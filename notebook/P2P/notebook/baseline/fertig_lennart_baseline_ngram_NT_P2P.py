# %% N-gram baseline — Next-Time (NT) on P2P
# - Temporal split by case start (same as your ACT code)
# - For each prefix of length k (k=1..n-1), target is Δt_k = t_k - t_{k-1} in days
# - n-gram stores mean Δt per context (last n-1 activities)
# - Backoff: (n-1)->(n-2)->...->unigram->global mean
# - Select best n via validation MAE; report per-k MAE/MSE/RMSE; W&B logging & plots

import os
os.environ["MPLBACKEND"] = "Agg"  # headless matplotlib

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from collections import defaultdict

import wandb
from sklearn import metrics

# %% W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

config = {
    "baseline":   "ngram_nt_mean",
    "n_values":   [2, 3, 4, 5, 6],   # candidate n for validation
    "min_coun":  3,                 # require >= min_coun for a context; else backoff
    "target_unit":"days",
}
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_ngram_NT_P2P",
    entity="privajet-university-of-mannheim",
    name=f"ngram_nt_{ts}",
    config=config,
    resume="never",
    force=True
)

# %% Data
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_p2p.csv.gz")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)

# Build NT prefixes (align with your NT/LSTM code):
# For case events at times t_0..t_{n-1}, for i in 1..n-1:
#   prefix = acts[:i]
#   next_time_delta = (t_i - t_{i-1}) in days
rows = []
for case_id, g in df.groupby("case:concept:name", sort=False):
    acts  = g["concept:name"].tolist()
    times = g["time:timestamp"].tolist()
    for i in range(1, len(acts)):
        rows.append({
            "case_id": case_id,
            "prefix": acts[:i],
            "next_time_delta": (times[i] - times[i-1]).total_seconds() / 86400.0,
            "k": i
        })
nt_df = pd.DataFrame(rows)

# Temporal split by case start
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

print(f"Train prefixes: {len(train_df)} - Val: {len(val_df)} - Test: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# Global mean (days) from TRAIN for ultimate fallback
GLOBAL_MEAN = float(train_df["next_time_delta"].mean()) if len(train_df) else float("nan")

# %% N-gram trainer (mean aggregation)
# Model format: dict[level_n][context_tuple] = {"sum": float, "count": int}
def fit_ngram_nt(df_part, n: int, min_coun: int):
    levels = {}  # build all levels 1..n for backoff (unigram to (n-1)-gram context)
    for L in range(1, n):  # context length = L = n-1,...,1 (we'll fill all; use backoff later)
        levels[L] = defaultdict(lambda: {"sum": 0.0, "count": 0})

    for _, r in df_part.iterrows():
        p = r["prefix"]
        y = float(r["next_time_delta"])
        # add contributions for all context sizes up to n-1 (so we can back off)
        max_ctx = min(len(p), n-1)
        for L in range(1, max_ctx + 1):
            ctx = tuple(p[-L:])
            node = levels[L][ctx]
            node["sum"] += y
            node["count"] += 1

    # Prune by min_coun
    for L in range(1, n):
        levels[L] = {ctx: sc for ctx, sc in levels[L].items() if sc["count"] >= min_coun}

    return levels  # levels[1]..levels[n-1]

def predict_delta(prefix, levels, n: int, global_mean: float):
    """Backoff from (n-1) to 1; return mean delta if seen else global mean."""
    max_ctx = min(len(prefix), n-1)
    for L in range(max_ctx, 0, -1):
        ctx = tuple(prefix[-L:])
        table = levels.get(L, {})
        sc = table.get(ctx)
        if sc is not None and sc["count"] > 0:
            return sc["sum"] / sc["count"]
    return global_mean

# %% Evaluation helpers
def eval_per_k_regression(df_part, levels, n: int, global_mean: float):
    k_vals, counts, maes, mses, rmses = [], [], [], [], []
    max_k = int(df_part["k"].max()) if len(df_part) else 0
    for k in range(1, max_k + 1):
        subset = df_part[df_part["k"] == k]
        if subset.empty:
            continue
        y_true = subset["next_time_delta"].values
        y_pred = np.array([predict_delta(p, levels, n, global_mean) for p in subset["prefix"]], dtype=np.float32)

        k_vals.append(k); counts.append(len(subset))
        mae  = metrics.mean_absolute_error(y_true, y_pred)
        mse  = metrics.mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        maes.append(mae); mses.append(mse); rmses.append(rmse)

    avg_mae  = float(np.mean(maes))  if maes  else float("nan")
    avg_mse  = float(np.mean(mses))  if mses  else float("nan")
    avg_rmse = float(np.mean(rmses)) if rmses else float("nan")
    return k_vals, counts, maes, mses, rmses, avg_mae, avg_mse, avg_rmse

# %% Model selection (by validation MAE; tie-break by RMSE)
best = {"n": None, "avg_mae": float("inf"), "avg_rmse": float("inf")}
for n in config["n_values"]:
    levels_val = fit_ngram_nt(train_df, n, min_coun=config["min_coun"])
    _, _, _, _, _, avg_mae, _, avg_rmse = eval_per_k_regression(val_df, levels_val, n, GLOBAL_MEAN)
    print(f"VAL n={n}: avg_mae={avg_mae:.4f} days, avg_rmse={avg_rmse:.4f} days")
    better = (avg_mae < best["avg_mae"]) or (np.isclose(avg_mae, best["avg_mae"]) and avg_rmse < best["avg_rmse"])
    if better:
        best.update({"n": n, "avg_mae": avg_mae, "avg_rmse": avg_rmse})

n_final = int(best["n"]) if best["n"] is not None else int(config["n_values"][0])
wandb.config.update({"n_selected": n_final}, allow_val_change=True)
print(f"Selected n={n_final} (val avg_mae={best['avg_mae']:.4f}, avg_rmse={best['avg_rmse']:.4f})")

# %% Fit final levels on TRAIN
levels = fit_ngram_nt(train_df, n_final, min_coun=config["min_coun"])

# %% Test evaluation
k_vals, counts, maes, mses, rmses, avg_mae, avg_mse, avg_rmse = \
    eval_per_k_regression(test_df, levels, n_final, GLOBAL_MEAN)

print(f"Average MAE across all prefixes:  {avg_mae:.2f} days")
print(f"Average MSE across all prefixes:  {avg_mse:.2f} (days^2)")
print(f"Average RMSE across all prefixes: {avg_rmse:.2f} days")

# %% Plots → disk
plot_dir = "/ceph/lfertig/Thesis/notebook/P2P/plots/Baselines/NGRAM/NT"
os.makedirs(plot_dir, exist_ok=True)

if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, maes, marker='o', label='MAE')
    plt.title(f'NT (n-gram mean) — MAE vs. Prefix Length (k), n={n_final}')
    plt.xlabel('Prefix Length (k)'); plt.ylabel('MAE (days)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mae_vs_k_n{n_final}_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, rmses, marker='o', label='RMSE')
    plt.title(f'NT (n-gram mean) — RMSE vs. Prefix Length (k), n={n_final}')
    plt.xlabel('Prefix Length (k)'); plt.ylabel('RMSE (days)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"rmse_vs_k_n{n_final}_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, mses, marker='o', label='MSE')
    plt.title(f'NT (n-gram mean) — MSE vs. Prefix Length (k), n={n_final}')
    plt.xlabel('Prefix Length (k)'); plt.ylabel('MSE (days^2)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mse_vs_k_n{n_final}_{ts}.png"), dpi=150); plt.close()

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
    "config/n_final": n_final,
    "global_mean_days_train": GLOBAL_MEAN,
})

# %% Global scatter + error histogram (days)
y_true_all = test_df["next_time_delta"].values if len(test_df) else np.array([])
y_pred_all = np.array([predict_delta(p, levels, n_final, GLOBAL_MEAN) for p in test_df["prefix"]], dtype=np.float32) \
             if len(test_df) else np.array([])
abs_err = np.abs(y_true_all - y_pred_all).reshape(-1)

if len(y_true_all):
    tab = wandb.Table(
        data=[[float(y_true_all[i]), float(y_pred_all[i]), float(abs_err[i])] for i in range(len(abs_err))],
        columns=["true_days", "pred_days", "abs_err_days"]
    )
    wandb.log({
        "scatter_true_vs_pred": wandb.plot.scatter(tab, "true_days", "pred_days", title="NT n-gram: True vs Pred (days)"),
        "error_hist": wandb.Histogram(abs_err),
    })

# %% Sample predictions (days)
sample = test_df.sample(n=min(5, len(test_df)), random_state=42) if len(test_df) else test_df
s_table = wandb.Table(columns=["case_id","k","prefix","gold_days","pred_days","abs_err_days"])
for _, r in sample.iterrows():
    pred = float(predict_delta(r["prefix"], levels, n_final, GLOBAL_MEAN))
    gold = float(r["next_time_delta"])
    s_table.add_data(r["case_id"], r["k"], " → ".join(r["prefix"]), gold, pred, abs(gold - pred))
    print("Prefix:", " → ".join(r["prefix"]))
    print(f"Gold (days): {gold:.2f}")
    print(f"Pred (days): {pred:.2f}")
    print("-"*60)
wandb.log({"samples": s_table})

# %% Save model (JSON-serializable)
# Convert contexts to strings: "A␟B␟C" (U+241F as unlikely separator) or just "||"
SEP = "||"
serializable = {}
for L, table in levels.items():
    serializable[str(L)] = {SEP.join(list(ctx)): {"sum": float(v["sum"]), "count": int(v["count"])}
                            for ctx, v in table.items()}

model_dir = f"/tmp/ngram_nt_{ts}"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"ngram_nt_n{n_final}.json")
with open(model_path, "w") as f:
    json.dump({
        "n": n_final,
        "min_coun": int(config["min_coun"]),
        "global_mean_days_train": GLOBAL_MEAN,
        "levels": serializable,
        "sep": SEP,
    }, f)

artifact = wandb.Artifact(name=f"ngram_nt_artifacts_{ts}", type="model")
artifact.add_file(model_path)
run.log_artifact(artifact)

# %%
wandb.finish()