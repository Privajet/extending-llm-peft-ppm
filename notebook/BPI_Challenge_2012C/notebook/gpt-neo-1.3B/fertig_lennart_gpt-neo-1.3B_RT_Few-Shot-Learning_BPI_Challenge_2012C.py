# %% Few-shot (ICL) Remaining-Time (RT) prediction with GPT-Neo-1.3B on BPI_Challenge_2012C
import os, sys, glob, ctypes

prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
lib_candidates = sorted(glob.glob(os.path.join(prefix, "lib", "libstdc++.so.6*")))
if lib_candidates:
    try:
        ctypes.CDLL(lib_candidates[0], mode=getattr(os, "RTLD_GLOBAL", 0))
    except OSError as e:
        print(f"WARN: could not preload {lib_candidates[0]}: {e}")

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    project="gpt-neo-1.3B_RT_Few-Shot_BPI_Challenge_2012C",
    entity="privajet-university-of-mannheim",
    name=f"neo_fewshot_rt_{ts}",
    config={"seed": SEED}
)

# %% Data
DF_PATH = "/ceph/lfertig/Thesis/data/processed/df_bpi_challenge.csv.gz"
df = pd.read_csv(DF_PATH)
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)
log.info("Data shape: %s", df.shape)
print(df.head(10))

# Build RT prefixes: predict minutes remaining until case END at each prefix
rows = []
for cid, g in df.groupby("case:concept:name", sort=False):
    acts  = g["concept:name"].tolist()
    times = g["time:timestamp"].tolist()
    case_end = times[-1]
    for i in range(1, len(acts)):  # prefixes of length i, remaining time from event i-1 to end
        rows.append({
            "case_id": cid,
            "prefix": acts[:i],
            "remaining_time": (case_end - times[i-1]).total_seconds() / 60.0,  # minutes
            "k": i
        })
rt_df = pd.DataFrame(rows)
print("Total prefix samples:", len(rt_df))

# Temporal split by case start (identical pattern as ACT few-shot)
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

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

maxlen = int(rt_df["k"].max()) if len(rt_df) else 0

# Fallback when decoding fails: median train remaining time (minutes)
FALLBACK_MINUTES = float(np.median(train_df["remaining_time"])) if len(train_df) else 0.0
log.info("Fallback (median train remaining minutes): %.2f", FALLBACK_MINUTES)

# %% Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_LEN = 512  # keep tail with 'Assistant: '

BASE_MODEL = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="right")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    low_cpu_mem_usage=True,
    torch_dtype=DTYPE,
).to(DEVICE).eval()

# %% Retrieval for few-shot demos (TF-IDF on TRAIN prefixes)
def seq_str(pfx):  # simple whitespace-joined text
    return " ".join(pfx)

train_df["prefix_str"] = train_df["prefix"].apply(seq_str)
tfidf       = TfidfVectorizer().fit(train_df["prefix_str"])
train_tfidf = tfidf.transform(train_df["prefix_str"])

N_SHOTS = 5
MAX_DEMO_EVENTS  = 10
MAX_QUERY_EVENTS = 20

def _fmt_minutes(x: float) -> str:
    # demos: integer minutes for stability
    return str(int(round(float(x))))

def retrieve_demos(prefix, n_shots=N_SHOTS):
    q    = tfidf.transform([seq_str(prefix)])
    sims = cosine_similarity(q, train_tfidf).ravel()
    topk = np.argsort(sims)[-n_shots*3:][::-1]  # over-sample; then dedup by target value
    demos, seen_vals = [], set()
    for ridx in topk:
        ex = train_df.iloc[ridx]
        y  = float(ex["remaining_time"])
        y_rounded = int(round(y))
        if y_rounded in seen_vals:
            continue
        p_demo = ex["prefix"][-MAX_DEMO_EVENTS:]
        demos.append((p_demo, y))
        seen_vals.add(y_rounded)
        if len(demos) >= n_shots:
            break
    return demos

# %% Prompt builder (numeric targets, like NT but for remaining-to-end)
SYSTEM_LINE = (
    "System: Predict the remaining time until the CASE COMPLETES in MINUTES. "
    "Reply with ONLY a non-negative number (no text, no units).\n"
)

def make_prompt_with_demos(demos, query_prefix):
    blocks = [SYSTEM_LINE]
    for p, y in demos:
        blocks.append(
            f"User: {' → '.join(p)}\n"
            f"Assistant: {_fmt_minutes(y)}\n"
        )
    q_short = query_prefix[-MAX_QUERY_EVENTS:]
    blocks.append(
        f"User: {' → '.join(q_short)}\n"
        "Assistant: "
    )
    return "".join(blocks)

# %% Generation + robust parsing
@torch.no_grad()
def predict_remaining_minutes_few_shot(prefix):
    demos  = retrieve_demos(prefix, n_shots=N_SHOTS)
    prompt = make_prompt_with_demos(demos, prefix)

    enc = tokenizer(
        prompt,
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
        max_new_tokens=12,        # enough for a number
        do_sample=False,          # greedy for stability
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )[0]

    new_tokens = gen_ids[input_ids.size(1):]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Extract first number; clamp to [0, 30 days]
    m = re.search(r"(\d+(\.\d+)?)", text)
    if not m:
        return FALLBACK_MINUTES
    minutes = float(m.group(1))
    if not np.isfinite(minutes):
        return FALLBACK_MINUTES
    minutes = float(np.clip(minutes, 0.0, 60.0 * 24.0 * 30.0))
    return minutes

# %% Per-k evaluation
k_vals, counts = [], []
maes, mses, rmses = [], [], []

for k in range(1, maxlen + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue

    y_true = subset["remaining_time"].values
    y_pred = np.array([predict_remaining_minutes_few_shot(p) for p in subset["prefix"]], dtype=np.float32)

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
y_pred_all = np.array([predict_remaining_minutes_few_shot(p) for p in test_df["prefix"]], dtype=np.float32)
abs_err = np.abs(y_true_all - y_pred_all)

tab = wandb.Table(
    data=[[float(y_true_all[i]), float(y_pred_all[i]), float(abs_err[i])] for i in range(len(abs_err))],
    columns=["true_min", "pred_min", "abs_err_min"]
)
wandb.log({
    "scatter_true_vs_pred": wandb.plot.scatter(tab, "true_min", "pred_min", title="RT (ICL): True vs Predicted (minutes)"),
    "error_hist": wandb.Histogram(abs_err),
})

# %% Plots → disk (headless)
plot_dir = "/ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/plots/ICL/RT_FewShot"
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

# %% Samples table
sample = test_df.sample(n=min(5, len(test_df)), random_state=SEED) if len(test_df) else test_df
s_table = wandb.Table(columns=["case_id","k","prefix","gold_min","pred_min","abs_err_min"])
for _, r in sample.iterrows():
    pred = predict_remaining_minutes_few_shot(r["prefix"])
    gold = float(r["remaining_time"])
    s_table.add_data(r["case_id"], r["k"], " → ".join(r["prefix"]), gold, pred, abs(gold - pred))
    print("Prefix:", " → ".join(r["prefix"]))
    print(f"Gold (min): {gold:.2f}")
    print(f"Pred (min): {pred:.2f}")
    print("-"*60)
wandb.log({"samples": s_table})

# %% Finish
wandb.finish()