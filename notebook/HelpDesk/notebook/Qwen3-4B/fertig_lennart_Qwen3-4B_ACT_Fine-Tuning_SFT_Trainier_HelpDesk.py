# %% Setup
import os, sys, glob, ctypes, random, logging
os.environ["MPLBACKEND"] = "Agg"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Preload libstdc++ for some HPC stacks
prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
cands = glob.glob(os.path.join(prefix, "lib", "libstdc++.so.6*"))
if cands:
    try:
        mode = getattr(os, "RTLD_GLOBAL", 0)
        ctypes.CDLL(cands[0], mode=mode)
    except OSError:
        pass

import numpy as np
import pandas as pd
import torch
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import List, Dict
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import Loraconfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTconfig

# %% 
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# %% 
DATASET = "HelpDesk"

config = {
    "dataset":                  DATASET,
    "plots_dir":                f"/ceph/lfertig/Thesis/notebook/{DATASET}/plots/Qwen3-4B/FT/ACT",
    "out_dir":                  f"/ceph/lfertig/Thesis/models/{DATASET}/Qwen3-4B/ACT/act_ft_{ts}"
}

FT_CFG = {
    # model / runtime
    "model_name":               "Qwen/Qwen3-4B-Instruct-2507",
    "dtype":                    "fp16",                                 # "fp32" on CPU
    "device":                   "auto",
    "event_sep":                " → ",
    # Chat system instruction to force single-label outputs
    "system_msg":               (
                                "You are an assistant for next-activity prediction."
                                "Given a trace and a label list, choose EXACTLY ONE label from the list below and output ONLY that label."
                                ),
    # prompt formatting (completion-style so we can mask loss cleanly)
    "prompt_tmpl_demo":         (
                                "Trace: {trace}\n"
                                "Labels:\n{labels}\n"
                                "Answer: {gold}\n\n"
                                ),
    "prompt_tmpl_query":        (
                                "Trace: {trace}\n"
                                "Labels:\n{labels}\n"
                                "Answer:"
                                ),
    # training
    "epochs":                   3,
    "micro_bsz":                1,
    "grad_accum":               8,
    "lr":                       2e-4,
    "warmup_ratio":             0.05,
    "max_seq_len":              768,                   # Qwen has long context; keep reasonable

    # LoRA
    "lora_r":                   16,
    "lora_alpha":               64,
    "lora_dropout":             0.05,

    # LoRA target modules (Qwen/LLama-style)
    "target_modules": [
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
    ],
}

# %%
config["seed"] = 41
random.seed(config["seed"]);
np.random.seed(config["seed"]); 
torch.manual_seed(config["seed"])
if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(config["seed"])

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)
log.info("PyTorch: %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available(): log.info("GPU: %s", torch.cuda.get_device_name(0))

run = wandb.init(
    project=f"Qwen3-4B_ACT_FineTuning_{config['dataset']}",
    entity="privajet-university-of-mannheim",
    name=f"qwen3-4b_ft_act_{ts}",
    config={"config": config, "FT_CFG": FT_CFG},
    resume="never"
)

# %% Data
train_df = pd.read_csv(f"/ceph/lfertig/Thesis/data/{config['dataset']}/processed/next_activity_train.csv")
val_df   = pd.read_csv(f"/ceph/lfertig/Thesis/data/{config['dataset']}/processed/next_activity_val.csv")
test_df  = pd.read_csv(f"/ceph/lfertig/Thesis/data/{config['dataset']}/processed/next_activity_test.csv")

for d in (train_df, val_df, test_df):
    d.rename(columns={"next_act": "next_activity"}, inplace=True)
    d["prefix"] = d["prefix"].astype(str).str.split()

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# Stable label space from all splits (so eval isn't surprised by unseen labels)
label_list = sorted(pd.concat([
    train_df["next_activity"], val_df["next_activity"], test_df["next_activity"]
]).unique())
labels_for_prompt = "\n".join(label_list)

# %% Tokenizer & Base Model
from transformers import BitsAndBytesconfig

MODEL_NAME = FT_CFG["model_name"]
DTYPE = (torch.float16 if (torch.cuda.is_available() and FT_CFG["dtype"]=="fp16") else torch.float32)
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and FT_CFG["device"]=="auto") else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

# You can enable 4-bit for memory if needed (set quantization_config=bnb_cfg below)
# bnb_cfg = BitsAndBytesconfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    # quantization_config=bnb_cfg,   # uncomment to 4-bit load
).to(DEVICE)

if hasattr(base_model.config, "use_cache"):
    base_model.config.use_cache = False  # must be off for training

# %% LoRA
from peft import Loraconfig, get_peft_model

peft_cfg = Loraconfig(
    r=FT_CFG["lora_r"],
    lora_alpha=FT_CFG["lora_alpha"],
    lora_dropout=FT_CFG["lora_dropout"],
    target_modules=FT_CFG["target_modules"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, peft_cfg)

# Optional: gradient checkpointing for memory
try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    model.gradient_checkpointing_enable()

# Ensure inputs require grad for some backends
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def _make_inputs_require_grad(module, inputs, output):
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

model.print_trainable_parameters()

# %% Build SFT dataset (completion-only after "Assistant:")
def df_to_text(df: pd.DataFrame, with_gold=True) -> List[str]:
    sep = config["event_sep"]
    rows = []
    for _, r in df.iterrows():
        trace = sep.join(r["prefix"])
        if with_gold:
            rows.append(FT_CFG["prompt_tmpl_demo"].format(
                trace=trace, labels=labels_for_prompt, gold=r["next_activity"]))
        else:
            rows.append(FT_CFG["prompt_tmpl_query"].format(
                trace=trace, labels=labels_for_prompt))
    return rows

ds = DatasetDict({
    "train": Dataset.from_dict({"text": df_to_text(train_df, with_gold=True)}),
    "validation": Dataset.from_dict({"text": df_to_text(val_df, with_gold=True)}),
})
len(ds["train"]), len(ds["validation"])

# %% Mask loss to completion tokens only (after the last "Assistant:")
from transformers import DataCollatorForLanguageModeling

class CompletionOnlyCollator:
    def __init__(self, tokenizer, anchor="Assistant:", max_length=FT_CFG["max_seq_len"]):
        self.tok = tokenizer
        self.max_length = max_length
        self.anchor_ids = tokenizer(anchor, add_special_tokens=False).input_ids

    @staticmethod
    def _find_last(seq, pat):
        for i in range(len(seq)-len(pat), -1, -1):
            if seq[i:i+len(pat)] == pat:
                return i
        return -1

    def _mask_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()
        B, T = input_ids.size()
        ar = torch.arange(T, device=input_ids.device)
        for i in range(B):
            ids = input_ids[i].tolist()
            j = self._find_last(ids, self.anchor_ids)
            labels[i].fill_(-100)
            if j != -1:
                cut = j + len(self.anchor_ids)
                keep = (attention_mask[i] == 1) & (ar >= cut)
                labels[i, keep] = input_ids[i, keep]
        return labels

    def __call__(self, features):
        batch = self.tok.pad(features, padding=True, return_tensors="pt")
        # left-truncate to max_length
        for key in ("input_ids", "attention_mask"):
            if batch[key].size(1) > self.max_length:
                batch[key] = batch[key][:, -self.max_length:]
        batch["labels"] = self._mask_labels(batch["input_ids"], batch["attention_mask"])
        return batch

collator = CompletionOnlyCollator(tokenizer)

# %% Trainer (TRL SFT)
from trl import SFTTrainer, SFTconfig

sft_cfg = SFTconfig(
    output_dir=config["out_dir"],
    num_train_epochs=FT_CFG["epochs"],
    learning_rate=FT_CFG["lr"],
    per_device_train_batch_size=FT_CFG["micro_bsz"],
    per_device_eval_batch_size=FT_CFG["micro_bsz"],
    gradient_accumulation_steps=FT_CFG["grad_accum"],
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=FT_CFG["warmup_ratio"],
    weight_decay=0.05,
    optim="adamw_torch",
    max_grad_norm=0.5,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    bf16=False,
    fp16=(FT_CFG["dtype"] == "fp16"),
    max_seq_length=FT_CFG["max_seq_len"],
    packing=False,
    dataset_text_field="text",
    report_to=["wandb"],
)

trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
)

trainer.train()
os.makedirs(config["out_dir"], exist_ok=True)
trainer.model.save_pretrained(config["out_dir"])
tokenizer.save_pretrained(config["out_dir"])
log.info("Saved LoRA adapters & tokenizer to %s", config["out_dir"])

if hasattr(trainer.model, "config") and hasattr(trainer.model.config, "use_cache"):
    trainer.model.config.use_cache = True

# %% Reload (base + adapters) for inference
base_infer = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=DTYPE
).to(DEVICE)
infer_model = PeftModel.from_pretrained(base_infer, config["out_dir"])
if hasattr(infer_model, "gradient_checkpointing_disable"):
    infer_model.gradient_checkpointing_disable()
infer_model.eval()
if hasattr(infer_model.config, "use_cache"):
    infer_model.config.use_cache = True

device = next(infer_model.parameters()).device

# %% Exact label scoring (log-likelihood of [prompt + label])
from torch.nn.utils.rnn import pad_sequence

LABEL_IDS = {lbl: tokenizer(" " + lbl, add_special_tokens=False).input_ids for lbl in label_list}

def _scores_for_labels(prefix: List[str], labels=label_list) -> np.ndarray:
    """Compute log-likelihood scores for continuing the query prompt with each label."""
    sep = config["event_sep"]
    seq = sep.join(prefix)
    prompt = FT_CFG["prompt_tmpl_query"].format(trace=seq, labels=labels_for_prompt) + " "
    P_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    rows, lens = [], []
    for lb in labels:
        L_ids = torch.tensor(LABEL_IDS[lb], dtype=torch.long)
        rows.append(torch.cat([P_ids, L_ids], dim=0))
        lens.append(len(L_ids))

    input_ids = pad_sequence(rows, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        logits = infer_model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    cut = P_ids.size(0)

    scores = []
    for i, lb in enumerate(labels):
        L = lens[i]
        lp = torch.log_softmax(logits[i, cut-1:cut-1+L, :], dim=-1)
        tgt = torch.tensor(LABEL_IDS[lb], device=lp.device)
        s = lp.gather(-1, tgt.unsqueeze(-1)).sum()
        # length-normalize to compare labels with differing tokenization
        s = s / max(1, L)
        scores.append(float(s))
    return np.array(scores, dtype=np.float32)

def _softmax_np(x: np.ndarray):
    x = x.astype(np.float32)
    x -= x.max()
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(x) / len(x)

def predict_topk(prefix: List[str], k=5):
    scores = _scores_for_labels(prefix, labels=label_list)
    probs = _softmax_np(scores)
    idx = np.argsort(probs)[-k:][::-1]
    pred_idx = int(np.argmax(probs))
    pred = label_list[pred_idx]
    return pred, [label_list[i] for i in idx], float(probs[pred_idx]), [float(probs[i]) for i in idx]

# %% Evaluation (per-k, macro across k, micro global)
k_vals, accuracies, fscores, precisions, recalls, counts = [], [], [], [], [], []

for k in sorted(test_df["k"].astype(int).unique()):
    subset = test_df[test_df["k"] == k]
    if subset.empty: 
        continue
    y_true = subset["next_activity"].tolist()
    prefixes = subset["prefix"].tolist()
    y_pred = [predict_topk(p, k=1)[0] for p in prefixes]

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    k_vals.append(k); counts.append(len(y_true))
    accuracies.append(acc); precisions.append(prec); recalls.append(rec); fscores.append(f1)

avg_accuracy = float(np.mean(accuracies)) if accuracies else float("nan")
avg_f1       = float(np.mean(fscores))    if fscores    else float("nan")
avg_precision= float(np.mean(precisions)) if precisions else float("nan")
avg_recall   = float(np.mean(recalls))    if recalls    else float("nan")

print(f"Average accuracy across k:  {avg_accuracy:.4f}")
print(f"Average f1 across k:        {avg_f1:.4f}")
print(f"Average precision across k: {avg_precision:.4f}")
print(f"Average recall across k:    {avg_recall:.4f}")

# Micro/global on VAL & TEST
y_true_val  = val_df["next_activity"].tolist()
y_pred_val  = [predict_topk(p, k=1)[0] for p in val_df["prefix"].tolist()]
micro_acc_val = accuracy_score(y_true_val, y_pred_val)
print(f"[VAL]  Micro accuracy: {micro_acc_val:.4f}")

y_true_test = test_df["next_activity"].tolist()
y_pred_test = [predict_topk(p, k=1)[0] for p in test_df["prefix"].tolist()]
micro_acc_test = accuracy_score(y_true_test, y_pred_test)
print(f"[TEST] Micro accuracy: {micro_acc_test:.4f}")

wandb.log({
    "metrics/avg_accuracy": avg_accuracy,
    "metrics/avg_f1": avg_f1,
    "metrics/avg_precision": avg_precision,
    "metrics/avg_recall": avg_recall,
    "metrics/micro_acc_val": micro_acc_val,
    "metrics/micro_acc_test": micro_acc_test,
    "curves/k": k_vals,
    "curves/counts": counts,
    "curves/accuracy": accuracies,
    "curves/f1": fscores,
    "curves/precision": precisions,
    "curves/recall": recalls,
})

# %% Plots
os.makedirs(config["plots_dir"], exist_ok=True)
if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, accuracies, marker="o", label="Accuracy")
    plt.title("Accuracy vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("Accuracy")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(config['plots_dir'], f"acc_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, fscores, marker="o", label="F1 (weighted)")
    plt.title("F1 vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("F1 (weighted)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(config['plots_dir'], f"f1_vs_k_{ts}.png"), dpi=150); plt.close()

print(f"Saved plots to: {config['plots_dir']}")

# %% Top-k (3/5) on entire test + samples table
def topk_accuracy(y_true, topk_list, k=3):
    hits = sum(y_true[i] in topk_list[i][:k] for i in range(len(y_true)))
    return hits / len(y_true) if len(y_true) else float("nan")

topk_all = [predict_topk(p, k=5)[1] for p in test_df["prefix"]]
y_all    = test_df["next_activity"].tolist()
wandb.log({
    "metrics/top3_acc": float(topk_accuracy(y_all, topk_all, k=3)),
    "metrics/top5_acc": float(topk_accuracy(y_all, topk_all, k=5)),
})

sample = test_df.sample(n=min(5, len(test_df)), random_state=SEED) if len(test_df) else test_df
table = wandb.Table(columns=["k", "prefix", "gold", "pred", "p_pred", "top5", "top5_p"])
for _, r in sample.iterrows():
    toks = r["prefix"] if isinstance(r["prefix"], list) else str(r["prefix"]).split()
    pred, top5, p_pred, top5_p = predict_topk(toks, k=5)
    prefix_pretty = config["event_sep"].join(toks)
    gold = str(r["next_activity"])
    print("Prefix:", prefix_pretty)
    print("Gold:  ", gold)
    print(f"Pred:   {pred} ({p_pred:.3f})")
    print("Top-5: ", top5)
    print("-"*60)
    table.add_data(
        int(r["k"]),
        prefix_pretty,
        gold,
        pred,
        float(p_pred),
        ", ".join(top5),
        ", ".join(f"{x:.3f}" for x in top5_p),
    )
wandb.log({"samples": table})

# %% Finish
wandb.finish()