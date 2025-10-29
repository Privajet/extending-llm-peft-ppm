# %% N-gram — Next-Time (NT) prediction
# - Temporal split by case start (uses processed splits from `data.loader`)
# - Each training prefix provides target Δt (next_time) in days
# - Model: mean Δt per context using activity n-grams (context = last n−1 activities)
# - Back-off: (n−1)-gram → ... → unigram → global mean
# - Model selection: choose n (from config.n_values) minimizing validation MAE
# - Evaluation: per-k MAE/MSE/RMSE (days) + macro averages; scatter + histogram; sample preds
# - Logging/plots: W&B logging + headless matplotlib

import os, sys, glob, ctypes
os.environ["MPLBACKEND"] = "Agg"  # headless matplotlib

# Preload libstdc++ on some HPC stacks (no-op if not needed)
prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
cands = glob.glob(os.path.join(prefix, "lib", "libstdc++.so.6*"))
if cands:
    try:
        mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        ctypes.CDLL(cands[0], mode=mode)
    except OSError:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from collections import defaultdict

import wandb
from sklearn import metrics

# Data Pipeline
from data import loader
from data.constants import Task

# %% W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %% Config
DATASET = "BPIC20_DomesticDeclarations"

config = {
    # bookkeeping
    "dataset":                  DATASET,
    "n_values":                 [2, 3, 4, 5, 6],   # candidates for n
    "min_count":                3,                 # prune contexts with < min_count observations
}

# %% Init
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project=f"baseline_ngram_NT_{config['dataset']}",
    entity="privajet-university-of-mannheim",
    name=f"ngram_nt_{ts}",
    config=config,
    resume="never",
    force=True
)

# %% Data
data_loader = loader.LogsDataLoader(name=config["dataset"])
(train_df, test_df, val_df,
 x_word_dict, y_word_dict,
 max_case_length, vocab_size,
 num_output) = data_loader.load_data(task=Task.NEXT_TIME)

wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# %% Helpers
def to_tokens(prefix_cell):
    # Prefixes are stored as space-joined strings in NT loader
    if isinstance(prefix_cell, list):
        return [str(t) for t in prefix_cell]
    return str(prefix_cell).strip().split()

def fit_ngram_nt(df_part: pd.DataFrame, n: int, min_count: int):
    # levels[L]: context (tuple of last L activities) -> {"sum": float, "count": int}
    levels = {L: defaultdict(lambda: {"sum": 0.0, "count": 0}) for L in range(1, n)}
    for _, r in df_part.iterrows():
        prefix_tokens = to_tokens(r["prefix"])
        y = float(r["next_time"])
        max_ctx = min(len(prefix_tokens), n - 1)
        for L in range(1, max_ctx + 1):
            ctx = tuple(prefix_tokens[-L:])
            node = levels[L][ctx]
            node["sum"] += y
            node["count"] += 1
    # prune by frequency
    for L in range(1, n):
        levels[L] = {ctx: sc for ctx, sc in levels[L].items() if sc["count"] >= min_count}
    return levels

def predict_delta(prefix_tokens, levels, n: int, global_mean: float):
    # Back-off from L = min(len(prefix), n-1) down to 1; fallback to global mean
    max_ctx = min(len(prefix_tokens), n - 1)
    for L in range(max_ctx, 0, -1):
        ctx = tuple(prefix_tokens[-L:])
        sc = levels.get(L, {}).get(ctx)
        if sc:
            return sc["sum"] / sc["count"]
    return global_mean

# %% Global mean (days)
GLOBAL_MEAN = float(train_df["next_time"].mean())

# %% Model selection: choose n minimizing validation MAE
best_n = None
best_levels = None
best_val_mae = np.inf

for n in config["n_values"]:
    lvls = fit_ngram_nt(train_df, n=n, min_count=config["min_count"])
    y_true = val_df["next_time"].to_numpy(dtype=float)
    y_pred = np.array(
        [predict_delta(to_tokens(p), lvls, n, GLOBAL_MEAN) for p in val_df["prefix"]],
        dtype=float
    )
    val_mae = metrics.mean_absolute_error(y_true, y_pred)
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_n = n
        best_levels = lvls

wandb.log({"model/n_final": best_n, "val_mae_best": float(best_val_mae)})
print(f"[VAL] Selected n={best_n} (MAE={best_val_mae:.4f} days)")

# %% TEST — per-k evaluation (days)
k_vals, counts, maes, mses, rmses = [], [], [], [], []

for k in sorted(test_df["k"].astype(int).unique()):
    subset = test_df[test_df["k"] == k]
    y_true = subset["next_time"].to_numpy(dtype=float)
    y_pred = np.array(
        [predict_delta(to_tokens(p), best_levels, best_n, GLOBAL_MEAN) for p in subset["prefix"]],
        dtype=float
    )
    mae  = metrics.mean_absolute_error(y_true, y_pred)
    mse  = metrics.mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    k_vals.append(int(k))
    counts.append(int(len(subset)))
    maes.append(float(mae)); mses.append(float(mse)); rmses.append(float(rmse))

# Macro averages across k
avg_mae  = float(np.mean(maes))
avg_mse  = float(np.mean(mses))
avg_rmse = float(np.mean(rmses))

print(f"[TEST] Average MAE:  {avg_mae:.3f} days")
print(f"[TEST] Average MSE:  {avg_mse:.3f} days^2")
print(f"[TEST] Average RMSE: {avg_rmse:.3f} days")

# %% Plots → disk (days)
plot_dir = f"/ceph/lfertig/Thesis/notebook/{config['dataset']}/plots/Baselines/NGRAM/NT"
os.makedirs(plot_dir, exist_ok=True)

plt.figure(figsize=(8,5))
plt.plot(k_vals, maes, marker='o', label='MAE (days)')
plt.title('NT — MAE vs. Prefix Length (k)')
plt.xlabel('Prefix Length (k)'); plt.ylabel('MAE (days)')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"mae_vs_k_{ts}.png"), dpi=150); plt.close()

plt.figure(figsize=(8,5))
plt.plot(k_vals, rmses, marker='o', label='RMSE (days)')
plt.title('NT — RMSE vs. Prefix Length (k)')
plt.xlabel('Prefix Length (k)'); plt.ylabel('RMSE (days)')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"rmse_vs_k_{ts}.png"), dpi=150); plt.close()

plt.figure(figsize=(8,5))
plt.plot(k_vals, mses, marker='o', label='MSE (days^2)')
plt.title('NT — MSE vs. Prefix Length (k)')
plt.xlabel('Prefix Length (k)'); plt.ylabel('MSE (days^2)')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"mse_vs_k_{ts}.png"), dpi=150); plt.close()

print(f"Saved plots to: {plot_dir}")

# %% Log curves + macro metrics to W&B
wandb.log({
    "curves/k": k_vals,
    "curves/counts": counts,
    "curves/mae": maes,
    "curves/mse": mses,
    "curves/rmse": rmses,
    "metrics/avg_mae":  avg_mae,
    "metrics/avg_mse":  avg_mse,
    "metrics/avg_rmse": avg_rmse,
    "global_mean_days_train": GLOBAL_MEAN,
})

# %% Global scatter + error histogram (days)
y_true_all = test_df["next_time"].to_numpy(dtype=float)
y_pred_all = np.array(
    [predict_delta(to_tokens(p), best_levels, best_n, GLOBAL_MEAN) for p in test_df["prefix"]],
    dtype=float
)
abs_err = np.abs(y_true_all - y_pred_all).reshape(-1)

tab = wandb.Table(
    data=[[float(y_true_all[i]), float(y_pred_all[i]), float(abs_err[i])]
          for i in range(len(abs_err))],
    columns=["true_days", "pred_days", "abs_err_days"]
)
wandb.log({
    "scatter_true_vs_pred": wandb.plot.scatter(tab, "true_days", "pred_days", title="NT n-gram: True vs Pred (days)"),
    "error_hist": wandb.Histogram(abs_err),
})

# %% Sample predictions (days)
sample = test_df.sample(n=min(5, len(test_df)), random_state=42)
s_table = wandb.Table(columns=["case_id","k","prefix","gold_days","pred_days","abs_err_days"])
for _, r in sample.iterrows():
    tokens = to_tokens(r["prefix"])
    pred = float(predict_delta(tokens, best_levels, best_n, GLOBAL_MEAN))
    gold = float(r["next_time"])
    prefix_pretty = " → ".join(tokens)
    print("Prefix:", prefix_pretty)
    print(f"Gold (days): {gold:.2f}")
    print(f"Pred (days): {pred:.2f}")
    print("-"*60)
    s_table.add_data(r.get("case_id", ""), int(r["k"]), prefix_pretty, gold, pred, abs(gold - pred))
wandb.log({"samples": s_table})

# %%
wandb.finish()