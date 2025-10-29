# %%
import os, sys, random, re, logging
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
import torch
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
import wandb

from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import peft, torch.nn as nn
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from trl.trainer import sft_trainer as _trl_sft_trainer
_trl_sft_trainer.wandb = wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import List, Dict

# %%
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

log.info("PyTorch: %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available():
    log.info("GPU: %s", torch.cuda.get_device_name(0))

# %%
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="gpt-neo-1.3B_ACT_Fine-Tuning_SFT_Trainer_BPI_Challenge_2012C",
    entity="privajet-university-of-mannheim",
    name=f"llm_act_{ts}",
)

# %%
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_bpi_challenge.csv.gz")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)
log.info("Data shape: %s", df.shape)
print(df.head(10))

# %%
records = []
for case_id, g in df.groupby("case:concept:name"):
    acts = g["concept:name"].tolist()
    for i in range(1, len(acts)):  # prefix length i, predict acts[i]
        records.append({
            "case_id": case_id,
            "prefix": acts[:i],
            "next_activity": acts[i],
            "k": i,
            "case_start": g["time:timestamp"].iloc[0],
        })
act_df = pd.DataFrame(records)
print("Total prefix samples:", len(act_df))

# %%

label_list = sorted(act_df["next_activity"].unique())
label2id   = {lbl:i for i,lbl in enumerate(label_list)}
id2label   = {i:lbl for lbl,i in label2id.items()}

labels_for_prompt = " | ".join(label_list)

print("Labels:", label_list)

# Record maximum prefix length for later loops
maxlen = act_df["k"].max()

# %%
case_start = (
    df.groupby("case:concept:name")["time:timestamp"]
      .min().reset_index().sort_values("time:timestamp")
)
valid_cases = set(act_df["case_id"])
case_ids = [c for c in case_start["case:concept:name"] if c in valid_cases]

n_total = len(case_ids)
n_train = int(n_total * 0.8)
n_val   = int(n_train * 0.2)

train_ids = case_ids[: n_train - n_val]
val_ids   = case_ids[n_train-n_val : n_train]
test_ids  = case_ids[n_train : ]

train_df = act_df[act_df["case_id"].isin(train_ids)].reset_index(drop=True)
val_df   = act_df[act_df["case_id"].isin(val_ids)  ].reset_index(drop=True)
test_df  = act_df[act_df["case_id"].isin(test_ids) ].reset_index(drop=True)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

# %%
# SFT string formatting (kept as in your training)
def make_text(prefix, gold):
    seq = " → ".join(prefix)
    return (
        "System: Predict the next activity. Reply with EXACTLY one label.\n"
        f"User: {seq}\n"
        f"Labels: [{labels_for_prompt}]\n"
        f"Assistant: {gold}"
    )

train_texts = [make_text(p, y) for p, y in zip(train_df["prefix"], train_df["next_activity"])]
val_texts   = [make_text(p, y) for p, y in zip(val_df["prefix"],   val_df["next_activity"])]

train_ds = Dataset.from_dict({"text": train_texts})
val_ds   = Dataset.from_dict({"text": val_texts})
ds = DatasetDict({"train": train_ds, "validation": val_ds})
print(ds)

# %%
# Tokenizer (GPT-Neo: ggf. kein PAD-Token vorhanden -> PAD = EOS).
# - padding_side wird am Ende "right" (rechts-Padding für causal LM).
# - truncation_side="left": bei zu langen Sequenzen wird vorne abgeschnitten
BASE_MODEL = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="right")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token  # no new token added

tokenizer.truncation_side = "left"

print("PAD token:", tokenizer.pad_token, "PAD id:", tokenizer.pad_token_id)

# %% # Basismodell laden (float16 auf GPU spart VRAM/zeit).
use_cuda = torch.cuda.is_available()
dtype = torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    low_cpu_mem_usage=True,
    torch_dtype=dtype,
)
if use_cuda:
    base_model.to("cuda")

# %% 
# LoRA/PEFT einrichten:
# - Nur Attention-Projektionen werden trainiert (Q/K/V/Out).
# - Rest bleibt eingefroren -> schnell & speichereffizient.
try:
    from peft.tuners.lora import mark_only_lora_as_trainable, LoraLayer
except Exception:
    mark_only_lora_as_trainable = None
    LoraLayer = tuple()

print("PEFT version:", peft.__version__)
target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
print("LoRA target modules:", target_modules)

# Make sure adapters are **not** created in inference mode
peft_cfg = LoraConfig(
    # r=8,
    r = 16,
    # lora_alpha=32,
    lora_alpha=64,  
    lora_dropout=0.05,
    # lora_dropout=0.10,
    target_modules=target_modules,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,           # <-- critical on some PEFT versions
)

# Attach LoRA
model = get_peft_model(base_model, peft_cfg)

# Nur LoRA-Parameter trainierbar schalten
if mark_only_lora_as_trainable is not None:
    mark_only_lora_as_trainable(model)
else:
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)

# Diverse sichere Toggles
if hasattr(model, "set_adapter"): model.set_adapter("default")
if hasattr(model, "enable_adapter_layers"): model.enable_adapter_layers()
if hasattr(model.config, "use_cache"): model.config.use_cache = False

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    # Fallback for older Transformers
    def _make_inputs_require_grad(module, input, output):
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

try: 
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError: 
    model.gradient_checkpointing_enable()
model.print_trainable_parameters()
assert any("lora_" in n and p.requires_grad for n,p in model.named_parameters())


# %%
# # TRL SFTTrainer-Konfiguration:
# - Effektive Batchgröße = MICRO_BS * GRAD_ACC (hier 1 * 8 ~ 8).
# - Cosine-Scheduler + Warmup stabilisieren Startphase.
# - packing=False: jede Zeile als eigene Sequenz.
MAX_LEN = 512     
MICRO_BS = 1      # small micro-batch
GRAD_ACC = 8      # effective batch ~8
EPOCHS = 4
LR = 1e-4
WARMUP   = 0.05
bnb_config = None
out_dir  = r"model/gpt-neo-1.3B/act"

sft_cfg = SFTConfig(
    output_dir=out_dir,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=MICRO_BS,
    per_device_eval_batch_size=MICRO_BS,
    gradient_accumulation_steps=GRAD_ACC,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP,
    weight_decay=0.00,
    optim="adamw_torch",
    max_grad_norm=0.5,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    bf16=False,                                # 30xx often no native bf16
    fp16= False,
    max_seq_length=MAX_LEN,
    packing=False,                              # pack small samples for speed
    dataset_text_field="text",
    report_to=["wandb"],
)

# Collator: "Completion-Only" -> Loss nur auf Tokens NACH "Assistant:".
# - _find_last_subseq: robust gegen mehrfaches "Assistant:" im Prompt.
# - Achtung: Konsistenter Anker-String ("Assistant:") zwischen Train/Infer.

class CompletionOnlyCollatorRobust:
    def __init__(self, tokenizer, response_template="Assistant:", max_length=256):
        self.tok = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.max_length = max_length
        self.rt_ids = tokenizer(response_template, add_special_tokens=False).input_ids

    @staticmethod
    def _find_last_subseq(seq: List[int], pat: List[int]) -> int:
        for i in range(len(seq) - len(pat), -1, -1):
            if seq[i:i+len(pat)] == pat:
                return i
        return -1

    def _mask_after_anchor(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()
        B, T = input_ids.size()
        for i in range(B):
            ids = input_ids[i].tolist()
            attn = attention_mask[i]
            j = self._find_last_subseq(ids, self.rt_ids)
            labels[i].fill_(-100)
            if j != -1:
                cutoff = j + len(self.rt_ids)
                visible = (attn == 1)
                tail = torch.arange(T, device=attention_mask.device) >= cutoff
                keep = visible & tail
                labels[i, keep] = input_ids[i, keep]
        return labels

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        if "text" in features[0]:
            texts = [f["text"] for f in features]
            batch = self.tok(
                texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            batch["labels"] = self._mask_after_anchor(batch["input_ids"], batch["attention_mask"])
            return batch

        to_pad = []
        for f in features:
            d = {}
            d["input_ids"] = torch.tensor(f["input_ids"], dtype=torch.long) if not isinstance(f["input_ids"], torch.Tensor) else f["input_ids"]
            if "attention_mask" in f:
                d["attention_mask"] = torch.tensor(f["attention_mask"], dtype=torch.long) if not isinstance(f["attention_mask"], torch.Tensor) else f["attention_mask"]
            to_pad.append(d)

        batch = self.tok.pad(to_pad, padding=True, return_tensors="pt")
        if batch["input_ids"].size(1) > self.max_length:
            batch["input_ids"]      = batch["input_ids"][:, :self.max_length]
            batch["attention_mask"] = batch["attention_mask"][:, :self.max_length]
        batch["labels"] = self._mask_after_anchor(batch["input_ids"], batch["attention_mask"])
        return batch

collator = CompletionOnlyCollatorRobust(tokenizer, response_template="Assistant:", max_length=MAX_LEN)


trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
)

# %%
m = model
m.train()
batch = next(iter(trainer.get_train_dataloader()))
batch = {k: v.to(m.device) for k, v in batch.items()}
loss = m(**batch).loss
loss.backward()
print("One-batch loss:", loss.detach().item())
m.zero_grad(set_to_none=True)

# %%
train_result = trainer.train()
print("Done training.")
save_dir = out_dir
trainer.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("Saved adapters & tokenizer to:", save_dir)

# %%
from peft import PeftModel
gen_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)
if gen_model.get_input_embeddings().num_embeddings != len(tokenizer):
    gen_model.resize_token_embeddings(len(tokenizer))
if use_cuda: gen_model.to("cuda")
gen_model = PeftModel.from_pretrained(gen_model, save_dir)
if hasattr(gen_model, "gradient_checkpointing_disable"): gen_model.gradient_checkpointing_disable()
gen_model.eval()
if use_cuda: gen_model = gen_model.to("cuda")
gen_model.config.use_cache = True

try:
    print("hf_device_map:", getattr(gen_model, "hf_device_map", None))
except: pass
p = next(gen_model.parameters())
print("first_param_device:", p.device, "dtype:", p.dtype, "use_cache:", getattr(gen_model.config, "use_cache", None))

# %%
device = next(gen_model.parameters()).device

# Pre-tokenize labels once (no special tokens)
LABEL_IDS = {lbl: tokenizer(lbl, add_special_tokens=False).input_ids for lbl in label_list}

# Prompt template identical to training, up to "Assistant:" (no gold)
PROMPT_TMPL = (
    "System: Predict the next activity. Reply with EXACTLY one label.\n"
    "User: {}\n"
    f"Labels: [{labels_for_prompt}]\n"
    "Assistant:"
)

def predict_next_scoring(prefix):
    seq = " → ".join(prefix)
    prompt = PROMPT_TMPL.format(seq) + " "
    P = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    with torch.inference_mode():
        out = gen_model(input_ids=P, use_cache=True)
    pkv = out.past_key_values

    scores = []
    for lb in label_list:
            L = torch.tensor(LABEL_IDS[lb], device=device).unsqueeze(0)
            logits = gen_model(input_ids=L, past_key_values=pkv, use_cache=True).logits
            lp = torch.log_softmax(logits[0], dim=-1)
            tgt = L[0]
            scores.append(float(lp.gather(-1, tgt.unsqueeze(-1)).sum()))
    scores = np.asarray(scores, dtype=np.float32)
    return label_list[int(scores.argmax())], scores

# %%
import numpy as np, time, statistics as stats
print("len(test_df):", len(test_df))
print("\nCounts per k:")
print(test_df["k"].value_counts().sort_index())

# prompt token length stats on 500 random test samples
sample_idx = np.random.default_rng(42).choice(len(test_df), size=min(500, len(test_df)), replace=False)
sample_prompts = [" → ".join(test_df.iloc[i]["prefix"]) for i in sample_idx]
tok_lens = [len(tokenizer(PROMPT_TMPL.format(p) + " ").input_ids) for p in sample_prompts]
print("\nPrompt token length stats on sample:",
      f"mean={np.mean(tok_lens):.1f}, p50={np.percentile(tok_lens,50):.0f}, p95={np.percentile(tok_lens,95):.0f}")

# microbenchmark scorer on 50 samples
bench_n = min(50, len(test_df))
bench_prompts = [test_df.iloc[i]["prefix"] for i in range(bench_n)]
t = []
for pfx in bench_prompts:
    t0 = time.time()
    _ = predict_next_scoring(pfx)
    t.append(time.time() - t0)
mean_s = float(np.mean(t)) if t else 0.0
med_s  = float(stats.median(t)) if t else 0.0
print(f"\nScoring microbenchmark: n={len(t)}, median_s={med_s:.3f}, mean_s={mean_s:.3f}")

def _softmax(x):
    x = np.array(x, dtype=np.float32); x -= x.max()
    e = np.exp(x); s = e.sum()
    return (e/s) if s > 0 else np.ones_like(x)/len(x)

def predict_with_topk(prefix, k=5):
    pred, scores = predict_next_scoring(prefix)
    probs = _softmax(scores)
    top_idx = np.argsort(probs)[-k:][::-1]
    top_lbl = [label_list[i] for i in top_idx]
    top_p   = [float(probs[i]) for i in top_idx]
    p_pred  = float(probs[label_list.index(pred)])
    return pred, top_lbl, p_pred, top_p

# Per-k arrays
k_vals, counts = [], []
accuracies, precisions, recalls, fscores = [], [], [], []

for k in range(1, maxlen + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue
    y_true = subset["next_activity"].tolist()
    y_pred = [predict_next_scoring(p)[0] for p in subset["prefix"]]

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    k_vals.append(k); counts.append(len(subset))
    accuracies.append(acc); precisions.append(prec); recalls.append(rec); fscores.append(f1)

# Macro means across k-bins (do NOT append to curves)
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

# %%
y_true_all = test_df["next_activity"].tolist()
y_pred_all = [predict_next_scoring(p)[0] for p in test_df["prefix"]]
cm_labels  = sorted(set(map(str, y_true_all)) | set(map(str, y_pred_all)))

try:
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=[str(x).strip() for x in y_true_all],
            preds=[str(x).strip() for x in y_pred_all],
            class_names=cm_labels
        )
    })
except Exception as e:
    log.warning("W&B CM failed: %s (skipping).", e)

# %%
plot_dir = "/ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/plots/FT/ACT"
os.makedirs(plot_dir, exist_ok=True)

if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, accuracies, marker="o", label="Accuracy")
    plt.title("Accuracy vs. Prefix Length (k)"); plt.xlabel("Prefix Length (k)"); plt.ylabel("Accuracy")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"acc_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, fscores, marker="o", label="F1 (weighted)")
    plt.title("F1 vs. Prefix Length (k)"); plt.xlabel("Prefix Length (k)"); plt.ylabel("F1 (weighted)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"f1_vs_k_{ts}.png"), dpi=150); plt.close()

print(f"Saved plots to: {plot_dir}")

# %%
sample = test_df.sample(n=min(5, len(test_df)), random_state=SEED) if len(test_df) else test_df
table = wandb.Table(columns=["k","prefix","gold","pred","p_pred","top5","top5_p"])
for _, r in sample.iterrows():
    pred, top5, p_pred, top5_p = predict_with_topk(r["prefix"], k=5)
    print("Prefix:", " → ".join(r["prefix"]))
    print("Gold:  ", r["next_activity"])
    print(f"Pred:  {pred} ({p_pred:.3f})")
    print("Top-5:", top5)
    print("-"*60)
    table.add_data(
        r["k"],
        " → ".join(r["prefix"]),
        r["next_activity"],
        pred,
        float(p_pred),
        ", ".join(top5),
        ", ".join([f"{x:.3f}" for x in top5_p])
    )
wandb.log({"samples": table})

# %%
artifact = wandb.Artifact(name=f"neo_act_lora_{ts}", type="model")
artifact.add_dir(out_dir)
wandb.log_artifact(artifact)

# %%
wandb.finish()