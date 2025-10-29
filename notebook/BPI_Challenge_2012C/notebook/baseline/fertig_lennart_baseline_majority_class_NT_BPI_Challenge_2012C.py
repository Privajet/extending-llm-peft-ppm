# %% Majority (global-mean) baseline — Next-Time (NT) on BPI_Challenge_2012C

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
    "baseline": "majority_nt_global_mean",
    "dataset": "BPI_Challenge_2012C",
    "target_unit": "days",
    "constant_choice": "mean",    # ["mean", "median"] — switch if you prefer median robustness
}
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_majority_NT_BPI_Challenge_2012C",
    entity="privajet-university-of-mannheim",
    name=f"majority_nt_{ts}",
    config=config,
    resume="never",
    force=True
)

# %% Data
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_bpi_challenge.csv.gz")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)

# Build NT prefixes: for events e_0..e_{n-1}, i in [0..n-2]
#   prefix = acts[:i+1]
#   next_time_delta = t_{i+1} - t_i  (in days)
rows = []
for case_id, g in df.groupby("case:concept:name", sort=False):
    acts  = g["concept:name"].tolist()
    times = g["time:timestamp"].tolist()
    for i in range(len(acts) - 1):
        rows.append({
            "case_id": case_id,
            "prefix": acts[:i+1],
            "next_time_delta": (times[i+1] - times[i]).total_seconds() / 86400.0,
            "k": i + 1
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

print(f"Train prefixes: {len(train_df)} - Validation prefixes: {len(val_df)} - Test prefixes: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# %% Majority constant (global)
if config["constant_choice"] == "median":
    constant_days = float(train_df["next_time_delta"].median()) if len(train_df) else float("nan")
else:
    constant_days = float(train_df["next_time_delta"].mean()) if len(train_df) else float("nan")

print(f"Global constant (train, days) [{config['constant_choice']}]: {constant_days:.4f}")
wandb.log({"constant_days_train": constant_days})

# %% Per-k evaluation (days)
k_vals, counts = [], []
maes, mses, rmses = [], [], []

max_k = int(test_df["k"].max()) if len(test_df) else 0

for k in range(1, max_k + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue

    y_true = subset["next_time_delta"].values
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

# %% Plots → disk (days)
plot_dir = "/ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/plots/Baselines/MAJ/NT"
os.makedirs(plot_dir, exist_ok=True)

if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, maes, marker="o", label="MAE")
    plt.title("MAE vs. Prefix Length (k)"); plt.xlabel("Prefix Length (k)"); plt.ylabel("MAE (days)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mae_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, rmses, marker="o", label="RMSE")
    plt.title("RMSE vs. Prefix Length (k)"); plt.xlabel("Prefix Length (k)"); plt.ylabel("RMSE (days)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"rmse_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, mses, marker="o", label="MSE")
    plt.title("MSE vs. Prefix Length (k)"); plt.xlabel("Prefix Length (k)"); plt.ylabel("MSE (days^2)")
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
y_true_all = test_df["next_time_delta"].values if len(test_df) else np.array([])
y_pred_all = np.full_like(y_true_all, constant_days, dtype=np.float32)
abs_err = np.abs(y_true_all - y_pred_all).reshape(-1)

if len(y_true_all):
    tab = wandb.Table(
        data=[[float(y_true_all[i]), float(y_pred_all[i]), float(abs_err[i])] for i in range(len(abs_err))],
        columns=["true_days", "pred_days", "abs_err_days"]
    )
    wandb.log({
        "scatter_true_vs_pred": wandb.plot.scatter(tab, "true_days", "pred_days", title="NT (Majority): True vs Pred (days)"),
        "error_hist": wandb.Histogram(abs_err),
    })

# %% Sample predictions (days)
sample = test_df.sample(n=min(5, len(test_df)), random_state=42) if len(test_df) else test_df
s_table = wandb.Table(columns=["case_id","k","prefix","gold_days","pred_days","abs_err_days"])
for _, r in sample.iterrows():
    gold = float(r["next_time_delta"])
    pred = float(constant_days)
    s_table.add_data(r["case_id"], r["k"], " → ".join(r["prefix"]), gold, pred, abs(gold - pred))
    print("Prefix:", " → ".join(r["prefix"]))
    print(f"Gold (days): {gold:.2f}")
    print(f"Pred (days): {pred:.2f}")
    print("-"*60)
wandb.log({"samples": s_table})

# %%
wandb.finish()