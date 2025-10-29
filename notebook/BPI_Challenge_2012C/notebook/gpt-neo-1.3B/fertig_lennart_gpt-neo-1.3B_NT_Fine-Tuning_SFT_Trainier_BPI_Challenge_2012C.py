# %% NT fine-tuning (SFT + LoRA) with GPT-Neo-1.3B on BPI_Challenge_2012C

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

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from trl.trainer import sft_trainer as _trl_sft_trainer
_trl_sft_trainer.wandb = wandb

from sklearn.metrics import mean_absolute_error, mean_squared_error

# %% Repro / logging
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)
log.info("PyTorch: %s | CUDA: %s", torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available(): log.info("GPU: %s", torch.cuda.get_device_name(0))

# %% W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="gpt-neo-1.3B_NT_Fine-Tuning_SFT_Trainer_BPI_Challenge_2012C",
    entity="privajet-university-of-mannheim",
    name=f"llm_nt_{ts}",
)

# %% Data
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_bpi_challenge.csv.gz")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)
log.info("Data shape: %s", df.shape)
print(df.head(10))

# Build NT prefixes: predict delta to next event (minutes)
records = []
for case_id, g in df.groupby("case:concept:name", sort=False):
    acts  = g["concept:name"].tolist()
    times = g["time:timestamp"].tolist()
    for i in range(len(acts) - 1):  # prefix len i+1, predict time to next event at i+1
        records.append({
            "case_id": case_id,
            "prefix": acts[:i+1],
            "next_time_delta": (times[i+1] - times[i]).total_seconds() / 60.0,
            "k": i+1,
            "case_start": times[0],
        })
nt_df = pd.DataFrame(records)
print("Total prefix samples:", len(nt_df))

# Temporal split by case start (same as ACT FT)
case_start = (
    df.groupby("case:concept:name")["time:timestamp"]
      .min().reset_index().sort_values("time:timestamp")
)
valid_cases = set(nt_df["case_id"])
case_ids = [c for c in case_start["case:concept:name"] if c in valid_cases]

n_total = len(case_ids)
n_train = int(n_total * 0.8)
n_val   = int(n_train * 0.2)

train_ids = case_ids[: n_train - n_val]
val_ids   = case_ids[n_train - n_val : n_train]
test_ids  = case_ids[n_train : ]

train_df = nt_df[nt_df["case_id"].isin(train_ids)].reset_index(drop=True)
val_df   = nt_df[nt_df["case_id"].isin(val_ids)].reset_index(drop=True)
test_df  = nt_df[nt_df["case_id"].isin(test_ids)].reset_index(drop=True)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
maxlen = int(nt_df["k"].max()) if len(nt_df) else 0

# Fallback if decoding fails: median train delta (minutes)
FALLBACK_MIN = float(np.median(train_df["next_time_delta"])) if len(train_df) else 0.0
log.info("Fallback (median train delta minutes): %.2f", FALLBACK_MIN)

# %% SFT string formatting (anchor on 'Assistant:'; numeric-only target)
ROUND_TARGET = True  # round minutes to nearest int for training targets

def make_text(prefix, gold_minutes):
    seq = " → ".join(prefix)
    y  = int(round(float(gold_minutes))) if ROUND_TARGET else float(gold_minutes)
    return (
        "System: Predict the time until the next event in MINUTES. "
        "Reply with ONLY a non-negative number (no text, no units).\n"
        f"User: {seq}\n"
        f"Assistant: {y}"
    )

train_texts = [make_text(p, y) for p, y in zip(train_df["prefix"], train_df["next_time_delta"])]
val_texts   = [make_text(p, y) for p, y in zip(val_df["prefix"],   val_df["next_time_delta"])]

train_ds = Dataset.from_dict({"text": train_texts})
val_ds   = Dataset.from_dict({"text": val_texts})
ds = DatasetDict({"train": train_ds, "validation": val_ds})
print(ds)

# %% Tokenizer (same setup as your ACT FT)
BASE_MODEL = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token  # reuse eos as pad

tokenizer.padding_side = "right"
tokenizer.truncation_side = "left"

print("PAD token:", tokenizer.pad_token, "PAD id:", tokenizer.pad_token_id)

# %% Base model + PEFT (LoRA)
use_cuda = torch.cuda.is_available()
dtype = torch.float16 if use_cuda else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    low_cpu_mem_usage=True,
    torch_dtype=dtype,
)
if use_cuda: base_model.to("cuda")

import peft, torch.nn as nn
from peft import LoraConfig, get_peft_model
try:
    from peft.tuners.lora import mark_only_lora_as_trainable, LoraLayer
except Exception:
    mark_only_lora_as_trainable = None
    LoraLayer = tuple()

print("PEFT version:", peft.__version__)
target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
print("LoRA target modules:", target_modules)

peft_cfg = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=target_modules,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
)
model = get_peft_model(base_model, peft_cfg)
if mark_only_lora_as_trainable is not None:
    mark_only_lora_as_trainable(model)
else:
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)

if hasattr(model, "set_adapter"): model.set_adapter("default")
if hasattr(model, "enable_adapter_layers"): model.enable_adapter_layers()
if hasattr(model.config, "use_cache"): model.config.use_cache = False
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
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

# %% TRL SFTTrainer config (copied from ACT FT, adjusted output dir / project name)
MAX_LEN = 512
MICRO_BS = 1
GRAD_ACC = 8
EPOCHS   = 4
LR       = 3e-4
WARMUP   = 0.05
out_dir  = "model/gpt-neo-1.3B/nt"

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
    weight_decay=0.05,
    optim="adamw_torch",
    max_grad_norm=0.5,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    bf16=False,
    fp16=False,
    max_seq_length=MAX_LEN,
    packing=False,
    dataset_text_field="text",
    report_to=["wandb"],
)

# %% Completion-only collator (same as ACT FT; masks loss before 'Assistant:')
from typing import List, Dict

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
                tail = torch.arange(T) >= cutoff
                keep = visible & tail
                labels[i, keep] = input_ids[i, keep]
        return labels

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        batch = self.tok(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
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

# %% Quick sanity: 1 forward/backward
m = model
m.train()
batch = next(iter(trainer.get_train_dataloader()))
batch = {k: v.to(m.device) for k, v in batch.items()}
loss = m(**batch).loss
loss.backward()
print("One-batch loss:", loss.detach().item())
m.zero_grad(set_to_none=True)

# %% Train + save
train_result = trainer.train()
print("Done training.")
save_dir = out_dir
trainer.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("Saved adapters & tokenizer to:", save_dir)

# %% Reload base + adapters for generation inference
from peft import PeftModel
gen_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)
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

device = gen_model.device

# %% Prompt for inference (no gold, ends with 'Assistant: ')
PROMPT_TMPL = (
    "System: Predict the time until the next event in MINUTES. "
    "Reply with ONLY a non-negative number (no text, no units).\n"
    "User: {trace}\n"
    "Assistant: "
)

def _make_prompt(prefix_tokens):
    trace = " → ".join(prefix_tokens)
    return PROMPT_TMPL.format(trace=trace)

@torch.no_grad()
def predict_delta_minutes(prefix_tokens):
    """Greedy numeric generation + robust parsing with fallback and clamping."""
    prompt = _make_prompt(prefix_tokens)
    enc = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    gen_ids = gen_model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=12,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )[0]

    new_tokens = gen_ids[input_ids.size(1):]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    m = re.search(r"(\d+(\.\d+)?)", text)
    if not m:
        return FALLBACK_MIN
    minutes = float(m.group(1))
    if not np.isfinite(minutes):
        return FALLBACK_MIN
    # clamp to [0, 30 days]
    minutes = float(np.clip(minutes, 0.0, 60.0 * 24.0 * 30.0))
    return minutes

# %% Per-k evaluation
k_vals, counts = [], []
maes, mses, rmses = [], [], []

for k in range(1, maxlen + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue
    y_true = subset["next_time_delta"].values
    y_pred = np.array([predict_delta_minutes(p) for p in subset["prefix"]], dtype=np.float32)

    k_vals.append(k); counts.append(len(subset))
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    maes.append(mae); mses.append(mse); rmses.append(rmse)

# Macro averages
avg_mae  = float(np.mean(maes))  if maes  else float("nan")
avg_mse  = float(np.mean(mses))  if mses  else float("nan")
avg_rmse = float(np.mean(rmses)) if rmses else float("nan")

print(f"Average MAE across all prefixes:  {avg_mae:.2f}")
print(f"Average MSE across all prefixes:  {avg_mse:.2f}")
print(f"Average RMSE across all prefixes: {avg_rmse:.2f}")

wandb.log({
    "curves/k": k_vals,
    "curves/counts": counts,
    "curves/mae": maes,
    "curves/mse": mses,
    "curves/rmse": rmses,
    "metrics/avg_mae": avg_mae,
    "metrics/avg_mse": avg_mse,
    "metrics/avg_rmse": avg_rmse,
})

# %% Plots → disk
plot_dir = "/ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/plots/FT/NT"
os.makedirs(plot_dir, exist_ok=True)

if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, maes, marker="o", label="MAE")
    plt.title("MAE vs. Prefix Length (k)"); plt.xlabel("Prefix Length (k)"); plt.ylabel("MAE (minutes)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mae_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, rmses, marker="o", label="RMSE")
    plt.title("RMSE vs. Prefix Length (k)"); plt.xlabel("Prefix Length (k)"); plt.ylabel("RMSE (minutes)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"rmse_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, mses, marker="o", label="MSE")
    plt.title("MSE vs. Prefix Length (k)"); plt.xlabel("Prefix Length (k)"); plt.ylabel("MSE")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mse_vs_k_{ts}.png"), dpi=150); plt.close()

print(f"Saved plots to: {plot_dir}")

# %% Samples table (like ACT FT)
sample = test_df.sample(n=min(5, len(test_df)), random_state=SEED) if len(test_df) else test_df
table = wandb.Table(columns=["case_id","k","prefix","gold_min","pred_min","abs_err_min"])
for _, r in sample.iterrows():
    pred = predict_delta_minutes(r["prefix"])
    gold = float(r["next_time_delta"])
    table.add_data(
        r["case_id"], r["k"], " → ".join(r["prefix"]),
        gold, pred, abs(gold - pred)
    )
    print("Prefix:", " → ".join(r["prefix"]))
    print(f"Gold (min): {gold:.2f}")
    print(f"Pred (min): {pred:.2f}")
    print("-"*60)
wandb.log({"samples": table})

# %% Save adapters as W&B artifact
artifact = wandb.Artifact(name=f"neo_nt_lora_{ts}", type="model")
artifact.add_dir(out_dir)
wandb.log_artifact(artifact)

# %% Finish
wandb.finish()