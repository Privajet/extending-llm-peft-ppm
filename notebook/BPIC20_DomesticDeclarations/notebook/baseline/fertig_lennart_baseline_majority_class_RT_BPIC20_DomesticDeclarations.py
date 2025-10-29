# %% Majority (global-mean) baseline — Remaining Time (RT) on BPIC20_DomesticDeclarations
# - Temporal split by case start
# - For each prefix of length k (k=1..n-1), remaining_time = case_end - time_at_prefix_end (in days)
# - Predict constant = mean (or median) RT from TRAIN
# - Per-k MAE/MSE/RMSE (days), W&B logging, plots, samples

import os
os.environ["MPLBACKEND"] = "Agg"  # headless matplotlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
import wandb
from sklearn import metrics

# %% W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

config = {
    "baseline": "majority_rt_global_mean",
    "dataset": "BPIC20_DomesticDeclarations",
    "target_unit": "days",
    "constant_choice": "mean",   # ["mean", "median"]
}
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_majority_RT_BPIC20_DomesticDeclarations",
    entity="privajet-university-of-mannheim",
    name=f"majority_rt_{ts}",
    config=config,
    resume="never",
    force=True
)

# %% Data
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_bpic20_domestic.csv.gz")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)

#%%
# Build RT prefixes:
# For case events at times t_0..t_{n-1}, end = t_{n-1}.
# For each i in 1..n-1: prefix = acts[:i], remaining_time = end - t_{i-1}.
rows = []
for case_id, g in df.groupby("case:concept:name", sort=False):
    acts  = g["concept:name"].tolist()
    times = g["time:timestamp"].tolist()
    end_time = times[-1]
    for i in range(1, len(times)):  # prefix length i, use time index i-1
        rows.append({
            "case_id": case_id,
            "prefix": acts[:i],
            "remaining_time": (end_time - times[i-1]).total_seconds() / 86400.0,  # days
            "k": i
        })
rt_df = pd.DataFrame(rows)

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

train_df = rt_df[rt_df["case_id"].isin(train_ids)].reset_index(drop=True)
val_df   = rt_df[rt_df["case_id"].isin(val_ids)].reset_index(drop=True)
test_df  = rt_df[rt_df["case_id"].isin(test_ids)].reset_index(drop=True)

print(f"Train prefixes: {len(train_df)} - Val: {len(val_df)} - Test: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# %% Majority constant (global RT on TRAIN)
if config["constant_choice"] == "median":
    constant_days = float(train_df["remaining_time"].median()) if len(train_df) else float("nan")
else:
    constant_days = float(train_df["remaining_time"].mean()) if len(train_df) else float("nan")

print(f"Global constant RT (train, days) [{config['constant_choice']}]: {constant_days:.4f}")
wandb.log({"constant_rt_days_train": constant_days})

# %% Per-k evaluation (days)
k_vals, counts = [], []
maes, mses, rmses = [], [], []

max_k = int(test_df["k"].max()) if len(test_df) else 0

for k in range(1, max_k + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue

    y_true = subset["remaining_time"].values
    y_pred = np.full_like(y_true, constant_days, dtype=np.float32)

    k_vals.append(k); counts.append(len(subset))
    mae  = metrics.mean_absolute_error(y_true, y_pred)
    mse  = metrics.mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    maes.append(mae); mses.append(mse); rmses.append(rmse)

# Macro averages across k
avg_mae  = float(np.mean(maes))  if maes  else float("nan")
avg_mse  = float(np.mean(mses))  if mses  else float("nan")
avg_rmse = float(np.mean(rmses)) if rmses else float("nan")

print(f"Average MAE across all prefixes:  {avg_mae:.2f} days")
print(f"Average MSE across all prefixes:  {avg_mse:.2f} (days^2)")
print(f"Average RMSE across all prefixes: {avg_rmse:.2f} days")

# %% Plots → disk
plot_dir = "/ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/plots/Baselines/MAJ/RT"
os.makedirs(plot_dir, exist_ok=True)

if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, maes, marker="o", label="MAE")
    plt.title("RT — MAE vs. Prefix Length (k)"); plt.xlabel("Prefix Length (k)"); plt.ylabel("MAE (days)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mae_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, rmses, marker="o", label="RMSE")
    plt.title("RT — RMSE vs. Prefix Length (k)"); plt.xlabel("Prefix Length (k)"); plt.ylabel("RMSE (days)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"rmse_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, mses, marker="o", label="MSE")
    plt.title("RT — MSE vs. Prefix Length (k)"); plt.xlabel("Prefix Length (k)"); plt.ylabel("MSE (days^2)")
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
})

# %% Global scatter + error histogram (days)
y_true_all = test_df["remaining_time"].values if len(test_df) else np.array([])
y_pred_all = np.full_like(y_true_all, constant_days, dtype=np.float32)
abs_err = np.abs(y_true_all - y_pred_all).reshape(-1)

if len(y_true_all):
    tab = wandb.Table(
        data=[[float(y_true_all[i]), float(y_pred_all[i]), float(abs_err[i])] for i in range(len(abs_err))],
        columns=["true_days", "pred_days", "abs_err_days"]
    )
    wandb.log({
        "scatter_true_vs_pred": wandb.plot.scatter(tab, "true_days", "pred_days", title="RT (Majority): True vs Pred (days)"),
        "error_hist": wandb.Histogram(abs_err),
    })

# %% Sample predictions (days)
sample = test_df.sample(n=min(5, len(test_df)), random_state=42) if len(test_df) else test_df
s_table = wandb.Table(columns=["case_id","k","prefix","gold_days","pred_days","abs_err_days"])
for _, r in sample.iterrows():
    gold = float(r["remaining_time"])
    pred = float(constant_days)
    s_table.add_data(r["case_id"], r["k"], " → ".join(r["prefix"]), gold, pred, abs(gold - pred))
    print("Prefix:", " → ".join(r["prefix"]))
    print(f"Gold (days): {gold:.2f}")
    print(f"Pred (days): {pred:.2f}")
    print("-"*60)
wandb.log({"samples": s_table})

# %%
wandb.finish()