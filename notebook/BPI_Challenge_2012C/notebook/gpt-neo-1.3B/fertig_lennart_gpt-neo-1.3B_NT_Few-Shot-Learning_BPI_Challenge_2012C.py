# %% Few-shot (ICL) Next-Time (NT) prediction with GPT-Neo-1.3B on BPI_Challenge_2012C
import os, sys, glob, ctypes

# Prefer the active conda env's libstdc++.so.6 (has the required CXXABI_1.3.15+)
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
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")  # double-safety; respects MPLBACKEND=Agg
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
    project="gpt-neo-1.3B_NT_Few-Shot_BPI_Challenge_2012C",
    entity="privajet-university-of-mannheim",
    name=f"neo_fewshot_nt_{ts}",
    config={"seed": SEED}
)

# %% Data (BPI Challenge 2012C)
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_bpi_challenge.csv.gz")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)
log.info("Data shape: %s", df.shape)
print(df.head(10))

# Build NT prefixes: predict time delta (minutes) to next event
rows = []
for cid, g in df.groupby("case:concept:name", sort=False):
    acts  = g["concept:name"].tolist()
    times = g["time:timestamp"].tolist()
    for i in range(len(acts) - 1):
        rows.append({
            "case_id": cid,
            "prefix": acts[:i+1],
            "next_time_delta": (times[i+1] - times[i]).total_seconds() / 60.0,  # minutes
            "k": i + 1
        })
nt_df = pd.DataFrame(rows)
print("Total prefix samples:", len(nt_df))

# Temporal split by case start (identical to your ACT few-shot)
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

train_df = nt_df[nt_df["case_id"].isin(train_ids)].reset_index(drop=True)
val_df   = nt_df[nt_df["case_id"].isin(val_ids)].reset_index(drop=True)
test_df  = nt_df[nt_df["case_id"].isin(test_ids)].reset_index(drop=True)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

maxlen = int(nt_df["k"].max()) if len(nt_df) else 0

# Fallback when decoding fails: median train delta (minutes)
FALLBACK_MINUTES = float(np.median(train_df["next_time_delta"])) if len(train_df) else 0.0
log.info("Fallback (median train delta, minutes): %.2f", FALLBACK_MINUTES)

# %% Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_LEN = 1024  

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

# %% Retrieval for few-shot demos (TF-IDF on train prefixes)
def seq_str(pfx):
    return " ".join(pfx)

train_df["prefix_str"] = train_df["prefix"].apply(seq_str)
tfidf       = TfidfVectorizer().fit(train_df["prefix_str"])
train_tfidf = tfidf.transform(train_df["prefix_str"])

N_SHOTS = 8
MAX_DEMO_EVENTS  = 10
MAX_QUERY_EVENTS = 20

def _fmt_minutes(x: float) -> str:
    return str(int(round(float(x))))

def retrieve_demos(prefix, n_shots=N_SHOTS):
    q    = tfidf.transform([seq_str(prefix)])
    sims = cosine_similarity(q, train_tfidf).ravel()
    topk = np.argsort(sims)[-n_shots*3:][::-1]  # over-sample; then dedup by target value
    demos, seen_vals = [], set()
    for ridx in topk:
        ex = train_df.iloc[ridx]
        y  = float(ex["next_time_delta"])
        y_rounded = int(round(y))
        if y_rounded in seen_vals:
            continue
        p_demo = ex["prefix"][-MAX_DEMO_EVENTS:]
        demos.append((p_demo, y))
        seen_vals.add(y_rounded)
        if len(demos) >= n_shots:
            break
    return demos

# %% Discretize minutes + candidate miner (per last activity), from TRAIN only
train_df["next_min_int"] = (
    train_df["next_time_delta"].round().astype(int).clip(0, int(60*24*30))
)

mins_by_last = defaultdict(Counter)
for _, r in train_df.iterrows():
    last = r["prefix"][-1] if len(r["prefix"]) else "<START>"
    mins_by_last[last][int(r["next_min_int"])] += 1

global_min_hist = Counter(train_df["next_min_int"])

def candidate_minutes(prefix, K=20):
    last = prefix[-1] if len(prefix) else "<START>"
    c = [m for m, _ in mins_by_last[last].most_common(K)]
    if not c:
        c = [m for m, _ in global_min_hist.most_common(K)]
    return sorted(set(c))

# %% Prompt base (plain completion, no chat roles) for scoring
def make_prompt_base(prefix, demos):
    blocks = []
    for p, y in demos:
        blocks.append(
            f"Events: {' | '.join(p[-MAX_DEMO_EVENTS:])}\n"
            f"Minutes: {_fmt_minutes(y)}\n\n"
        )
    q_short = prefix[-MAX_QUERY_EVENTS:]
    blocks.append(f"Events: {' | '.join(q_short)}\nMinutes:")
    return "".join(blocks)

def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float32)
    x -= x.max()
    e = np.exp(x)
    s = e.sum()
    return e/s if s > 0 else np.ones_like(x)/len(x)

# %% Score numeric candidates via PLL (no generation)
@torch.no_grad()
def score_numeric_candidates(prefix, K=20):
    demos = retrieve_demos(prefix, n_shots=N_SHOTS)
    base  = make_prompt_base(prefix, demos)
    # Add one trailing space so number tokens are not prefixed with a BPE space token
    base_ids = tokenizer(base + " ", add_special_tokens=False, truncation=True, max_length=MAX_LEN).input_ids

    cands = candidate_minutes(prefix, K=K)
    rows, lens, cand_ids = [], [], []
    for m in cands:
        ids = tokenizer(str(int(m)), add_special_tokens=False).input_ids
        cand_ids.append(ids)
        rows.append(torch.tensor(base_ids + ids, dtype=torch.long))
        lens.append(len(ids))

    pad = tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad).to(DEVICE)
    attn = (input_ids != pad).long()
    logits = model(input_ids=input_ids, attention_mask=attn).logits.float()

    cut = len(base_ids)
    scores = []
    for i in range(len(cands)):
        L = lens[i]
        lp  = torch.log_softmax(logits[i, cut-1:cut-1+L, :], dim=-1)
        tgt = torch.tensor(cand_ids[i], device=attn.device)
        scores.append(float(lp.gather(-1, tgt.unsqueeze(-1)).sum()))
    return cands, np.array(scores, dtype=np.float32)

@torch.no_grad()
def predict_next_minutes_pruned(prefix, K=20, use_expectation=True, topk=5):
    try:
        cands, scores = score_numeric_candidates(prefix, K=K)
        probs = _softmax(scores)
        if use_expectation:
            pred = float(np.dot(probs, np.array(cands, dtype=np.float32)))
        else:
            pred = float(cands[int(np.argmax(scores))])
        # top-k for inspection
        idx = np.argsort(probs)[-min(topk, len(cands)):][::-1]
        top_vals = [int(cands[i]) for i in idx]
        top_p    = [float(probs[i]) for i in idx]
        return float(np.clip(pred, 0.0, 60.0*24.0*30.0)), top_vals, top_p
    except Exception as e:
        log.warning("Pruned prediction failed (%s); using fallback.", e)
        return FALLBACK_MINUTES, [], []

# %% Per-k evaluation with pruned predictor
K_PRUNE = 20  # tune {12,20,32}
k_vals, counts = [], []
maes, mses, rmses = [], [], []

for k in range(1, maxlen + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue

    y_true = subset["next_time_delta"].values
    preds = []
    for p in subset["prefix"]:
        pr, _, _ = predict_next_minutes_pruned(p, K=K_PRUNE, use_expectation=True, topk=5)
        preds.append(pr)
    y_pred = np.array(preds, dtype=np.float32)

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
    "prune/K": K_PRUNE,
})

# %% Global scatter + error histogram
y_true_all = test_df["next_time_delta"].values
y_pred_all = np.array([predict_next_minutes_pruned(p, K=K_PRUNE, use_expectation=True, topk=5)[0]
                       for p in test_df["prefix"]], dtype=np.float32)
abs_err = np.abs(y_true_all - y_pred_all)

tab = wandb.Table(
    data=[[float(y_true_all[i]), float(y_pred_all[i]), float(abs_err[i])] for i in range(len(abs_err))],
    columns=["true_min", "pred_min", "abs_err_min"]
)
wandb.log({
    "scatter_true_vs_pred": wandb.plot.scatter(tab, "true_min", "pred_min", title="NT (ICL, pruned): True vs Predicted (minutes)"),
    "error_hist": wandb.Histogram(abs_err),
})

# %% Plots → disk (headless)
plot_dir = "/ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/plots/ICL/NT_Pruned"
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

# %% Sample table (top-k candidates for inspection)
sample = test_df.sample(n=min(5, len(test_df)), random_state=SEED) if len(test_df) else test_df
s_table = wandb.Table(columns=["case_id","k","prefix","gold_min","pred_min","top5","top5_p","abs_err_min"])
for _, r in sample.iterrows():
    pred, top5, top5_p = predict_next_minutes_pruned(r["prefix"], K=K_PRUNE, use_expectation=True, topk=5)
    gold = float(r["next_time_delta"])
    s_table.add_data(
        r["case_id"],
        r["k"],
        " → ".join(r["prefix"]),
        gold,
        pred,
        ", ".join(map(str, top5)),
        ", ".join([f"{p:.3f}" for p in top5_p]),
        abs(gold - pred)
    )
    print("Prefix:", " → ".join(r["prefix"]))
    print(f"Gold (min): {gold:.2f}")
    print(f"Pred (min): {pred:.2f}")
    print(f"Top5: {top5}   p: {[f'{p:.3f}' for p in top5_p]}")
    print("-"*60)
wandb.log({"samples": s_table})

# %% Finish
wandb.finish()