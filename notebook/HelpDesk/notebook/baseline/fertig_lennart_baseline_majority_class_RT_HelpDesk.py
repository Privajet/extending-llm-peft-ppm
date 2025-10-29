# %% Majority (global-mean/median) baseline — Remaining Time (RT) on HelpDesk
# - Temporal split by case start (same splits as your other scripts)
# - For each prefix of length k, RT_k = t_end - t_{k-1} in days
# - Predict a single constant (mean/median) RT computed on TRAIN
# - Report per-k MAE/MSE/RMSE in days + macro averages; W&B logging; headless plots

import os, sys, random, glob, ctypes
os.environ["MPLBACKEND"] = "Agg"  # headless matplotlib

# Preload libstdc++ (HPC stacks)
prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
cands = glob.glob(os.path.join(prefix, "lib", "libstdc++.so.6*"))
if cands:
    try:
        mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        ctypes.CDLL(cands[0], mode=mode)
    except OSError:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
import wandb
from sklearn import metrics

# Data Pipeline
from data import loader
from data.constants import Task

# %% W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %% Config
config = {
    "dataset":          "HelpDesk",
    "constant_choice":  "median"  # or "mean"
}

# %%
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project=f"baseline_majority_RT_{config['dataset']}",
    entity="privajet-university-of-mannheim",
    name=f"majority_rt_{ts}",
    config=config,
    resume="never",
    force=True
)

# %% Data
data_loader = loader.LogsDataLoader(name=config['dataset'])
(train_df, test_df, val_df,
 x_word_dict, y_word_dict,
 max_case_length, vocab_size,
 num_output) = data_loader.load_data(task=Task.REMAINING_TIME)

wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# %% Majority constant
train_days = train_df["remaining_time_days"].to_numpy(dtype=float) if len(train_df) else np.array([])
if config["constant_choice"] == "median":
    constant_days = float(np.nanmedian(train_days)) if train_days.size else float("nan")
else:
    constant_days = float(np.nanmean(train_days)) if train_days.size else float("nan")

print(f"Global constant RT (train, days) [{config['constant_choice']}]: {constant_days:.4f}")
wandb.log({"constant_rt_days_train": constant_days})

# %% Per-k evaluation (days)
k_vals, counts, maes, mses, rmses = [], [], [], [], []

for k in sorted(test_df["k"].astype(int).unique()):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue
    
    y_true = subset["remaining_time_days"].to_numpy(dtype=float)
    y_pred = np.full_like(y_true, constant_days, dtype=float)

    mae  = metrics.mean_absolute_error(y_true, y_pred)
    mse  = metrics.mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    k_vals.append(int(k))
    counts.append(len(subset))
    maes.append(float(mae)); mses.append(float(mse)); rmses.append(float(rmse))

# Macro averages across k
avg_mae  = float(np.mean(maes))  if maes  else float("nan")
avg_mse  = float(np.mean(mses))  if mses  else float("nan")
avg_rmse = float(np.mean(rmses)) if rmses else float("nan")

print(f"Average MAE across all prefixes:  {avg_mae:.2f} days")
print(f"Average MSE across all prefixes:  {avg_mse:.2f} (days^2)")
print(f"Average RMSE across all prefixes: {avg_rmse:.2f} days")

# %% Plots → disk
plot_dir = f"/ceph/lfertig/Thesis/notebook/{config['dataset']}/plots/Baselines/MAJ/RT"
os.makedirs(plot_dir, exist_ok=True)

if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, maes, marker='o', label='MAE (days)')
    plt.title("RT — MAE vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("MAE (days)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mae_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, rmses, marker='o', label='RMSE (days)')
    plt.title("RT — RMSE vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("RMSE (days)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"rmse_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, mses, marker='o', label='MSE (days^2)')
    plt.title("RT — MSE vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("MSE (days^2)")
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
y_true_all = test_df["remaining_time_days"].to_numpy(dtype=float) if len(test_df) else np.array([])
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
    gold = float(r["remaining_time_days"])
    pred = float(constant_days)
    prefix_pretty = " → ".join(r["prefix"]) if isinstance(r["prefix"], (list, tuple)) else " → ".join(str(r["prefix"]).split())
    s_table.add_data(r["case_id"], int(r["k"]), prefix_pretty, gold, pred, abs(gold - pred))
    print("Prefix:", prefix_pretty)
    print(f"Gold (days): {gold:.2f}")
    print(f"Pred (days): {pred:.2f}")
    print("-"*60)
wandb.log({"samples": s_table})

# %%
wandb.finish()