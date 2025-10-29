# %% Zero-shot (ICL-free) Remaining-Time prediction (RT) with GPT-Neo-1.3B

import os, sys, glob, ctypes

prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
candidates = sorted(glob.glob(os.path.join(prefix, "lib", "libstdc++.so.6*")))
if candidates:
    try:
        mode = getattr(os, "RTLD_GLOBAL", 0)  # make symbols globally visible
        ctypes.CDLL(candidates[0], mode=mode)
        # print("Preloaded libstdc++:", candidates[0])  # optional debug
    except OSError as e:
        print("WARN: could not preload libstdc++ from env:", e)

# Headless plotting + avoid torchvision import from HF
os.environ["MPLBACKEND"] = "Agg"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Standard imports
import re
import random
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# Safe matplotlib import (now uses the preloaded libstdc++)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

# %% Repro + logging
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)
log.info("PyTorch: %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available(): log.info("GPU: %s", torch.cuda.get_device_name(0))

# %% W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="gpt-neo-1.3B_RT_Zero-Shot_BPI_Challenge_2012C",
    entity="privajet-university-of-mannheim",
    name=f"neo_zeroshot_rt_{ts}",
    config={"seed": SEED}
)

# %% Data
DF_PATH = "/ceph/lfertig/Thesis/data/processed/df_bpi_challenge.csv.gz"
df = pd.read_csv(DF_PATH)
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)
log.info("Data shape: %s", df.shape)
print(df.head(10))

# Build RT prefixes: remaining minutes from current event to case end
rows = []
for cid, g in df.groupby("case:concept:name", sort=False):
    acts  = g["concept:name"].tolist()
    times = g["time:timestamp"].tolist()
    case_end = times[-1]
    # prefixes 1..(n-1); for prefix acts[:i], remaining time from event i-1 to end
    for i in range(1, len(acts)):
        rows.append({
            "case_id": cid,
            "prefix": acts[:i],
            "remaining_time": (case_end - times[i-1]).total_seconds() / 60.0,  # minutes
            "k": i
        })
rt_df = pd.DataFrame(rows)

# Temporal split by case start (identical to ACT)
case_start = (
    df.groupby("case:concept:name")["time:timestamp"]
      .min().reset_index().sort_values("time:timestamp")
)
case_ids = case_start["case:concept:name"].tolist()

n_total = len(case_ids)
n_train = int(n_total * 0.8)
n_val   = int(n_train * 0.2)

train_ids = case_ids[: n_train - n_val]
val_ids   = case_ids[n_train - n_val : n_train]
test_ids  = case_ids[n_train : ]

train_df = rt_df[rt_df["case_id"].isin(train_ids)].reset_index(drop=True)
val_df   = rt_df[rt_df["case_id"].isin(val_ids)].reset_index(drop=True)
test_df  = rt_df[rt_df["case_id"].isin(test_ids)].reset_index(drop=True)

print(f"Train prefixes: {len(train_df)} - Validation prefixes: {len(val_df)} - Test prefixes: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

maxlen = int(rt_df["k"].max()) if len(rt_df) else 0

# Fallback if model output is unparsable: median remaining time (minutes) on TRAIN
FALLBACK_MINUTES = float(np.median(train_df["remaining_time"])) if len(train_df) else 0.0
log.info("Fallback (median train remaining, minutes): %.2f", FALLBACK_MINUTES)

# %% Model (GPU fp16 if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_LEN = 512  # keep the tail with 'Assistant: '

BASE_MODEL = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="right")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"  # preserve the 'Assistant: ' tail

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    low_cpu_mem_usage=True,
    torch_dtype=DTYPE,
).to(DEVICE).eval()

# %% Zero-shot prompt + inference
PROMPT_TMPL = (
    "System: You are a process mining assistant. Given a trace of activities from a running case, "
    "predict the remaining time until case completion in MINUTES. Reply with ONLY a non-negative number.\n"
    "User: {trace}\n"
    "Assistant: "
)

def _make_prompt(prefix_tokens):
    trace = " → ".join(prefix_tokens)
    return PROMPT_TMPL.format(trace=f"[{trace}]")

@torch.no_grad()
def predict_remaining_minutes(prefix_tokens):
    """
    Greedy-generate a short numeric answer (minutes).
    Robustly parse the first number; fallback to dataset median if none found.
    """
    enc = tokenizer(
        _make_prompt(prefix_tokens),
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attn_mask = enc["attention_mask"].to(DEVICE)

    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=10,         # enough for a number
        do_sample=False,           # greedy for stability
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )[0]

    new_tokens = gen_ids[input_ids.size(1):]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Extract first number (allow decimal); clamp to [0, 30 days]
    m = re.search(r"(\d+(\.\d+)?)", text)
    if not m:
        return FALLBACK_MINUTES
    minutes = float(m.group(1))
    if not np.isfinite(minutes):
        return FALLBACK_MINUTES
    minutes = float(np.clip(minutes, 0.0, 60.0 * 24.0 * 30.0))
    return minutes

# %% Per-k evaluation (MAE/MSE/RMSE)
k_vals, counts = [], []
maes, mses, rmses = [], [], []

for k in range(1, maxlen + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue

    y_true = subset["remaining_time"].values
    y_pred = np.array([predict_remaining_minutes(p) for p in subset["prefix"]], dtype=np.float32)

    k_vals.append(k); counts.append(len(subset))
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    maes.append(mae); mses.append(mse); rmses.append(rmse)

# Macro averages across k-bins
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

# %% Global scatter + error histogram
y_true_all = test_df["remaining_time"].values
y_pred_all = np.array([predict_remaining_minutes(p) for p in test_df["prefix"]], dtype=np.float32)
abs_err = np.abs(y_true_all - y_pred_all)

tab = wandb.Table(
    data=[[float(y_true_all[i]), float(y_pred_all[i]), float(abs_err[i])] for i in range(len(abs_err))],
    columns=["true_min", "pred_min", "abs_err_min"]
)
wandb.log({
    "scatter_true_vs_pred": wandb.plot.scatter(tab, "true_min", "pred_min", title="RT: True vs Predicted (minutes)"),
    "error_hist": wandb.Histogram(abs_err),
})

# %% Plots → disk (headless)
plot_dir = "/ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/plots/ZS/RT"
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

# %% Samples table (same schema style you use)
sample = test_df.sample(n=min(5, len(test_df)), random_state=SEED) if len(test_df) else test_df
s_table = wandb.Table(columns=["case_id","k","prefix","gold_min","pred_min","abs_err_min"])
for _, r in sample.iterrows():
    pred = predict_remaining_minutes(r["prefix"])
    gold = float(r["remaining_time"])
    s_table.add_data(r["case_id"], r["k"], " → ".join(r["prefix"]), gold, pred, abs(gold - pred))
    print("Prefix:", " → ".join(r["prefix"]))
    print(f"Gold (min): {gold:.2f}")
    print(f"Pred (min): {pred:.2f}")
    print("-"*60)
wandb.log({"samples": s_table})

# %% Finish
wandb.finish()