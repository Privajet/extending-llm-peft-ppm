# %% Fine-tuned Next-Activity (ACT) with GPT-Neo-1.3B — LoRA + SFT
# - Supervised fine-tuning via TRL SFT on prompts with gold labels (completion-only loss after "Assistant:")
# - Lightweight PEFT (LoRA) on GPT-Neo attention/projection layers (c_attn, c_proj) for GPU efficiency
# - Deterministic setup (seeds, no use_cache during train, gradient checkpointing)
# - Tokenizer left-truncation; fixed label vocabulary injected into the prompt
# - Training config: small micro-batch + grad accumulation; cosine schedule + warmup; fp16 if CUDA
# - Evaluation as generative multi-class: score each candidate label by log-likelihood of [prompt + label]
# - k-wise metrics (Accuracy, F1, Precision, Recall) over prefix length; plots saved + W&B logged
# - Artifacts: save LoRA adapters + tokenizer; reload PEFT for inference-time scoring

import os, sys, glob, ctypes, random, logging
os.environ["MPLBACKEND"]="Agg"
os.environ["TRANSFORMERS_NO_TORCHVISION"]="1"
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")

# Preload libstdc++ on some HPC stacks (no-op if not needed)
prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
cands = glob.glob(os.path.join(prefix, "lib", "libstdc++.so.6*"))
if cands:
    try:
        mode = getattr(os, "RTLD_GLOBAL", 0)
        ctypes.CDLL(cands[0], mode=mode)
    except OSError:
        pass

import numpy as np, pandas as pd, torch
from datetime import datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
from torch.nn.utils.rnn import pad_sequence

# %% Reproducibility & logging
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log=logging.getLogger("neo_act_ft")
log.info("PyTorch %s | CUDA %s", torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available(): log.info("GPU: %s", torch.cuda.get_device_name(0))

# %%  W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# %% Configs
RUN_CFG = {
    "seed": SEED,
    "case_col": "case:concept:name",
    "act_col":  "concept:name",
    "time_col": "time:timestamp",
    "plots_dir": "/ceph/lfertig/Thesis/notebook/HelpDesk/plots/gpt-neo-1.3B/FT/NT",
    "unit": "days",
}

FT_CFG = {
    # model / runtime
    "model_name": "EleutherAI/gpt-neo-1.3B",
    "dtype": "fp16",              # set "fp32" if CPU-only
    "device": "auto",
    # prompt & context
    "max_seq_len": 512,
    "event_sep": " → ",
    "prompt_tmpl_demo": (
        "System: Predict the next activity. Reply with EXACTLY one label.\n"
        "User: {trace}\n"
        "Labels:\n{labels}\n"
        "Assistant: {gold}"
    ),
    "prompt_tmpl_query": (
        "System: Predict the next activity. Reply with EXACTLY one label.\n"
        "User: {trace}\n"
        "Labels:\n{labels}\n"
        "Assistant:"
    ),
    "epochs": 3,
    "micro_bsz": 1,
    "grad_accum": 8,
    "lr": 3e-4,
    "warmup_ratio": 0.05,
    "lora_r": 16, 
    "lora_alpha": 64, 
    "lora_dropout": 0.05, 
    "out_dir": f"/ceph/lfertig/Thesis/models/gpt-neo-1.3B/ACT/act_ft_{ts}",
}

run = wandb.init(
    project="gpt-neo-1.3B_ACT_FineTuning_HelpDesk",
    entity="privajet-university-of-mannheim",
    name=f"neo_ft_act_{ts}",
    config={"run_cfg": RUN_CFG, "FT_CFG": FT_CFG},
)

# %% Data
train_df = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_train.csv")
val_df   = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_val.csv")
test_df  = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_test.csv")

for d in (train_df, val_df, test_df):
    if "next_act" in d.columns: d.rename(columns={"next_act":"next_activity"}, inplace=True)
    d["prefix"] = d["prefix"].astype(str).str.split()

log.info("Train=%d  Val=%d  Test=%d", len(train_df), len(val_df), len(test_df))
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

label_list = sorted(pd.concat([train_df["next_activity"], val_df["next_activity"], test_df["next_activity"]]).unique())
labels_for_prompt = "\n".join(label_list) # A single string listing labels, newline-separated, for prompt insertion

# %% Model / Tokenizer
MODEL_NAME = FT_CFG["model_name"]
DTYPE = (torch.float16 if (torch.cuda.is_available() and FT_CFG["dtype"]=="fp16") else torch.float32)
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and FT_CFG["device"]=="auto") else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True).to(DEVICE)
if hasattr(base_model.config, "use_cache"): base_model.config.use_cache=False

print("PAD token:", tokenizer.pad_token, "PAD id:", tokenizer.pad_token_id)

# %%
target_modules = ["c_attn", "c_proj"]
peft_cfg = LoraConfig(
    r=FT_CFG["lora_r"], lora_alpha=FT_CFG["lora_alpha"], lora_dropout=FT_CFG["lora_dropout"],
    target_modules=target_modules, bias="none", task_type="CAUSAL_LM", inference_mode=False
)
model = get_peft_model(base_model, peft_cfg)
try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    model.gradient_checkpointing_enable()

# Make inputs require grad for checkpointing (prevents your error)
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    # Fallback hook for older transformers
    def _make_inputs_require_grad(module, inputs, output):
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

# keep cache off during train
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

model.print_trainable_parameters()

def df_to_text(df: pd.DataFrame, with_gold=True):
    sep = FT_CFG.get("event_sep", " → ")
    rows=[]
    for _, r in df.iterrows():
        trace = sep.join(r["prefix"])
        if with_gold:
            rows.append(FT_CFG["prompt_tmpl_demo"].format(trace=trace, labels=labels_for_prompt, gold=r["next_activity"]))
        else:
            rows.append(FT_CFG["prompt_tmpl_query"].format(trace=trace, labels=labels_for_prompt))
    return rows

ds = DatasetDict({
    "train": Dataset.from_dict({"text": df_to_text(train_df, with_gold=True)}),
    "validation": Dataset.from_dict({"text": df_to_text(val_df, with_gold=True)}),
})

# Mask loss to completion only
class CompletionOnlyCollator:
    def __init__(self, tokenizer, anchor="Assistant:", max_length=FT_CFG["max_seq_len"]):
        self.tok = tokenizer
        self.max_length = max_length
        self.anchor_ids = tokenizer(anchor, add_special_tokens=False).input_ids

    @staticmethod
    def _find_last(seq, pat):
        for i in range(len(seq)-len(pat), -1, -1):
            if seq[i:i+len(pat)] == pat: return i
        return -1

    def _mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()
        B, T = input_ids.size()
        arange_T = torch.arange(T)
        for i in range(B):
            ids = input_ids[i].tolist()
            j = self._find_last(ids, self.anchor_ids)
            labels[i].fill_(-100)
            if j != -1:
                cut = j + len(self.anchor_ids)
                keep = (attention_mask[i] == 1) & (arange_T >= cut)
                labels[i, keep] = input_ids[i, keep]
        return labels

    def __call__(self, feats):
        # feats are already tokenized by TRL (have input_ids / attention_mask)
        batch = self.tok.pad(feats, padding=True, return_tensors="pt")
        if batch["input_ids"].size(1) > self.max_length:
            batch["input_ids"]      = batch["input_ids"][:, :self.max_length]
            batch["attention_mask"] = batch["attention_mask"][:, :self.max_length]
        batch["labels"] = self._mask(batch["input_ids"], batch["attention_mask"])
        return batch

collator = CompletionOnlyCollator(tokenizer)

# %%
sft_cfg = SFTConfig(
    output_dir=FT_CFG["out_dir"],
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
    fp16=True,
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

# %% Train & save
trainer.train()
save_dir=FT_CFG["out_dir"]
trainer.model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)
log.info("Saved adapters & tokenizer to %s", save_dir)

# %% Load for inference
gen_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, low_cpu_mem_usage=True)
gen_model.resize_token_embeddings(len(tokenizer)); gen_model.to(DEVICE)
gen_model = PeftModel.from_pretrained(gen_model, save_dir)
if hasattr(gen_model,"gradient_checkpointing_disable"): gen_model.gradient_checkpointing_disable()
gen_model.eval()
if hasattr(gen_model.config,"use_cache"): gen_model.config.use_cache=True
device = next(gen_model.parameters()).device

# %% Scoring & Evaluation
LABEL_IDS = {lbl: tokenizer(lbl, add_special_tokens=False).input_ids for lbl in label_list}

def _scores_for_labels(prefix, labels=label_list):
    """Return raw per-label log-likelihood scores for continuing the prompt with each label."""
    sep = FT_CFG.get("event_sep", " → ")
    seq = sep.join(prefix)
    prompt = (FT_CFG["prompt_tmpl_query"].format(trace=seq, labels=labels_for_prompt)) + " "
    P_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    rows, lens = [], []
    for lb in labels:
        L_ids = torch.tensor(LABEL_IDS[lb], dtype=torch.long)
        rows.append(torch.cat([P_ids, L_ids], dim=0))
        lens.append(len(L_ids))

    input_ids = pad_sequence(rows, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        logits = gen_model(input_ids=input_ids, attention_mask=attention_mask).logits

    cut = P_ids.size(0)
    scores = []
    for i, lb in enumerate(labels):
        L = lens[i]
        lp = torch.log_softmax(logits[i, cut-1:cut-1+L, :], dim=-1)
        tgt = torch.tensor(LABEL_IDS[lb], device=lp.device)
        scores.append(float(lp.gather(-1, tgt.unsqueeze(-1)).sum()))
    return np.array(scores, dtype=np.float32)

def _softmax_np(x: np.ndarray):
    x = x.astype(np.float32)
    x -= x.max()
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(x) / len(x)

def predict_topk(prefix, k=5):
    """Return (pred_label, topk_labels, p_pred, topk_probs)."""
    scores = _scores_for_labels(prefix, labels=label_list)
    probs = _softmax_np(scores)
    top_idx = np.argsort(probs)[-k:][::-1]
    pred_idx = int(np.argmax(probs))
    pred_lbl = label_list[pred_idx]
    top_lbl = [label_list[i] for i in top_idx]
    top_p   = [float(probs[i]) for i in top_idx]
    p_pred  = float(probs[pred_idx])
    return pred_lbl, top_lbl, p_pred, top_p

def predict_next(prefix):
    """Convenience wrapper for top-1."""
    return predict_topk(prefix, k=1)[0]

# %% Per-k evaluation
k_vals, accuracies, fscores, precisions, recalls, counts = [], [], [], [], [], []

for k in sorted(test_df["k"].astype(int).unique()):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue
    y_true = subset["next_activity"].tolist()
    y_pred = [predict_topk(p, k=1)[0] for p in subset["prefix"]]
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    k_vals.append(k); counts.append(len(subset))
    accuracies.append(acc); precisions.append(prec); recalls.append(rec); fscores.append(f1)

avg_acc = float(np.mean(accuracies)) if accuracies else float("nan")
avg_f1  = float(np.mean(fscores))    if fscores    else float("nan")
avg_p   = float(np.mean(precisions)) if precisions else float("nan")
avg_r   = float(np.mean(recalls))    if recalls    else float("nan")

print(f"Average accuracy across all prefixes:  {avg_acc:.4f}")
print(f"Average f-score across all prefixes:   {avg_f1:.4f}")
print(f"Average precision across all prefixes: {avg_p:.4f}")
print(f"Average recall across all prefixes:    {avg_r:.4f}")

wandb.log({
    "curves/k": k_vals,
    "curves/counts": counts,
    "curves/accuracy": accuracies,
    "curves/f1": fscores,
    "curves/precision": precisions,
    "curves/recall": recalls,
    "metrics/avg_accuracy": avg_acc,
    "metrics/avg_f1": avg_f1,
    "metrics/avg_precision": avg_p,
    "metrics/avg_recall": avg_r,
})

# %% Top-k accuracy on the whole test set 
def topk_accuracy(y_true, topk_labels_list, k=3):
    hits = sum(y_true[i] in topk_labels_list[i][:k] for i in range(len(y_true)))
    return hits / len(y_true) if len(y_true) else float("nan")

topk_all = [predict_topk(p, k=5)[1] for p in test_df["prefix"]]
y_all    = test_df["next_activity"].tolist()
wandb.log({
    "metrics/top3_acc": float(topk_accuracy(y_all, topk_all, k=3)),
    "metrics/top5_acc": float(topk_accuracy(y_all, topk_all, k=5)),
})

# %% Plots
plot_dir = RUN_CFG["plots_dir"]
os.makedirs(plot_dir, exist_ok=True)

if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, accuracies, marker="o", label="Accuracy")
    plt.title("Accuracy vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("Accuracy")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"acc_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, fscores, marker="o", label="F1 (weighted)")
    plt.title("F1 vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("F1 (weighted)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"f1_vs_k_{ts}.png"), dpi=150); plt.close()

print(f"Saved plots to: {plot_dir}")

# %% Samples table
sep = FT_CFG.get("event_sep", " → ")
sample = test_df.sample(n=min(5, len(test_df)), random_state=SEED) if len(test_df) else test_df
table = wandb.Table(columns=["k","prefix","gold","pred","p_pred","top3","top3_p"])
for _, r in sample.iterrows():
    pred, topk, p_pred, topk_p = predict_topk(r["prefix"], k=3)
    prefix_str = sep.join(r["prefix"])
    gold = r["next_activity"]
    print("Prefix:", prefix_str)
    print("Gold:  ", gold)
    print(f"Pred:  {pred} ({p_pred:.3f})")
    print("Top-3:", topk)
    print("-"*60)
    table.add_data(
        int(r["k"]),
        prefix_str,
        gold,
        pred,
        float(p_pred),
        ", ".join(topk),
        ", ".join(f"{x:.3f}" for x in topk_p),
    )
wandb.log({"samples": table})

# %% Finish
wandb.finish()