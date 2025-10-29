# %% TabPFN — Next-Time (NT) prediction
# - Temporal split by case start (aligned with your other scripts)
# - For each prefix of length k (k=1..n-1), target is Δt_k = t_k - t_{k-1} in days
# - Features = fixed-length padded sequence of activity IDS from metadata (PAD=0, UNK=1), using pre-trunc + pre-pad
# - Primary: TabPFNRegressor (if available). Fallback: TabPFNClassifier on quantile-binned LOG targets
#   → converts class probs to LOG-days via expected value, then expm1 back to days
# - W&B logging + per-k MAE/MSE/RMSE curves, scatter, samples, artifact

import os, sys, random, logging, glob, ctypes, json, joblib, torch
os.environ["MPLBACKEND"] = "Agg"   # headless matplotlib

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
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
import wandb

from sklearn import metrics

from tabpfn import TabPFNRegressor
    
# Data Pipeline
from data import loader
from data.constants import Task

# %% Repro/Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# %% Weights & Biases
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %% Config
DATASET = "HelpDesk"

config = {
    # bookkeeping
    "dataset":                  DATASET,
    # model scale
    "sample_size":              10000,  # downsample train for speed/stability; set None to disable
}

# %%
config["seed"] = 41
tf.keras.utils.set_random_seed(config["seed"])
tf.config.experimental.enable_op_determinism()
random.seed(config["seed"])
np.random.seed(config["seed"])

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project=f"baseline_tabpfn_NT_{config['dataset']}",
    entity="privajet-university-of-mannheim",
    name=f"tabpfn_NT_{ts}",
    config=config,
    resume="never",
    force=True
)

# %% 
data_loader = loader.LogsDataLoader(name=config['dataset'])

(train_df, test_df, val_df,
 x_word_dict, y_word_dict,
 max_case_length, vocab_size,
 num_output) = data_loader.load_data(task=Task.NEXT_TIME)

wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

(train_tok_x, train_time_x, train_y, time_scaler, y_scaler) = data_loader.prepare_data_next_time(train_df, x_word_dict, max_case_length)
(val_tok_x, val_time_x, val_y, _, _) = data_loader.prepare_data_next_time(val_df, x_word_dict, max_case_length, time_scaler=time_scaler, y_scaler=y_scaler, shuffle=False)
(test_tok_x, test_time_x, test_y, _, _) = data_loader.prepare_data_next_time(test_df, x_word_dict, max_case_length, time_scaler=time_scaler, y_scaler=y_scaler, shuffle=False)

# %%
# Merge token ids (N, L) and time feats (N, 3) into one tabular matrix (float32)
X_train = np.concatenate([train_tok_x, train_time_x], axis=1).astype(np.float32)
X_val   = np.concatenate([val_tok_x,   val_time_x],   axis=1).astype(np.float32)
X_test  = np.concatenate([test_tok_x,  test_time_x],  axis=1).astype(np.float32)

# Targets: use the standardized log-space target returned by the loader (shape (N,1) or (N,))
y_train = train_y.reshape(-1)
y_val   = val_y.reshape(-1)
y_test  = test_y.reshape(-1)

# %% Train TabPFN model
model = TabPFNRegressor(device=str(DEVICE), ignore_pretraining_limits=True)
for key in ("inference_batch_size","inference_max_batch_size","batch_size_inference"):
    try:
        model.set_params(**{key: 256}); break
    except Exception:
        pass

if config["sample_size"] and len(X_train) > config["sample_size"]:
    rng = np.random.default_rng(config["seed"])
    idx = rng.choice(len(X_train), size=config["sample_size"], replace=False)
    X_fit, y_fit = X_train[idx], y_train[idx]
    wandb.log({"train_downsampled_to": int(len(X_fit))})
else:
    X_fit, y_fit = X_train, y_train

model.fit(X_fit, y_fit)

# %%
def predict_days_from_features(X):
    y_scaled = model.predict(X).reshape(-1, 1)
    y_log    = y_scaler.inverse_transform(y_scaled)[:, 0]
    y_days   = np.expm1(y_log)
    return np.maximum(y_days, 0.0)

# %% Inference helper (returns days)
def predict_delta_days(prefix_str: str, recent_time=0.0, latest_time=0.0, time_passed=0.0) -> float:
    df1 = pd.DataFrame([{
        "prefix": prefix_str,
        "recent_time": float(recent_time),
        "latest_time": float(latest_time),
        "time_passed": float(time_passed),
        "k": 0
    }])
    tok_x, time_x, _, _, _ = data_loader.prepare_data_next_time(
        df1, x_word_dict, max_case_length,
        time_scaler=time_scaler, y_scaler=y_scaler, shuffle=False
    )
    X = np.concatenate([tok_x, time_x], axis=1).astype(np.float32)
    return float(predict_days_from_features(X)[0])

# %% Test per-k eval (days)
k_vals, maes, mses, rmses, counts = [], [], [], [], []

for k in range(1, int(max_case_length) + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue

    sub_tok_x, sub_time_x, sub_y, _, _ = data_loader.prepare_data_next_time(
        subset, x_word_dict, max_case_length,
        time_scaler=time_scaler, y_scaler=y_scaler, shuffle=False
    )

    X_k = np.concatenate([sub_tok_x, sub_time_x], axis=1).astype(np.float32)
    
    y_pred_scaled = model.predict(X_k).reshape(-1, 1)
    y_true_days   = y_scaler.inverse_transform(sub_y)
    y_pred_days   = y_scaler.inverse_transform(y_pred_scaled)

    mae  = metrics.mean_absolute_error(y_true_days, y_pred_days)
    mse  = metrics.mean_squared_error(y_true_days, y_pred_days)
    rmse = float(np.sqrt(mse))

    k_vals.append(k)
    counts.append(len(subset))
    maes.append(mae); mses.append(mse); rmses.append(rmse)

# Macro averages across k-bins
avg_mae  = float(np.mean(maes))  if maes  else float("nan")
avg_mse  = float(np.mean(mses))  if mses  else float("nan")
avg_rmse = float(np.mean(rmses)) if rmses else float("nan")

print(f"Average MAE across all prefixes:  {avg_mae:.2f} days")
print(f"Average MSE across all prefixes:  {avg_mse:.2f} (days^2)")
print(f"Average RMSE across all prefixes: {avg_rmse:.2f} days")

# %%  Plots
plot_dir = f"/ceph/lfertig/Thesis/notebook/{config['dataset']}/plots/Baselines/TabPFN/NT"
os.makedirs(plot_dir, exist_ok=True)

# Per-k (days)
if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, maes, marker='o', label='MAE (days)')
    plt.title('MAE vs. Prefix Length (k)')
    plt.xlabel('Prefix Length (k)'); plt.ylabel('MAE (days)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mae_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, rmses, marker='o', label='RMSE (days)')
    plt.title('RMSE vs. Prefix Length (k)'); plt.xlabel('Prefix Length (k)'); plt.ylabel('RMSE (days)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"rmse_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, mses, marker='o', label='MSE (days^2)')
    plt.title('MSE vs. Prefix Length (k)')
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
})

# %% Global scatter + error histogram (days)
y_pred_scaled_all = model.predict(X_test).reshape(-1, 1)
y_true_all_days   = y_scaler.inverse_transform(test_y)
y_pred_all_days   = y_scaler.inverse_transform(y_pred_scaled_all)
abs_err = np.abs(y_true_all_days - y_pred_all_days).reshape(-1)

tab = wandb.Table(
    data=[[float(y_true_all_days[i,0]), float(y_pred_all_days[i,0]), float(abs_err[i])]
          for i in range(len(abs_err))],
    columns=["true_days", "pred_days", "abs_err_days"]
)
wandb.log({
    "scatter_true_vs_pred": wandb.plot.scatter(tab, "true_days", "pred_days", title="NT: True vs Pred (days)"),
    "error_hist": wandb.Histogram(abs_err),
})

# %% Sample predictions (days)
sample = test_df.sample(n=min(5, len(test_df)), random_state=42) if len(test_df) else test_df
table = wandb.Table(columns=["case_id","k","prefix","gold_days","pred_days","abs_err_days"])

for _, r in sample.iterrows():
    sub = r.to_frame().T
    sub_tok_x, sub_time_x, sub_y, _, _ = data_loader.prepare_data_next_time(
        sub, x_word_dict, max_case_length,
        time_scaler=time_scaler, y_scaler=y_scaler, shuffle=False
    )
    X_sub = np.concatenate([sub_tok_x, sub_time_x], axis=1).astype(np.float32)
    pred_scaled = model.predict(X_sub).reshape(-1, 1)
    gold_days = float(y_scaler.inverse_transform(sub_y)[0, 0])
    pred_days = float(y_scaler.inverse_transform(pred_scaled)[0, 0])
    
    print("Prefix:", " → ".join(r["prefix"].split() if isinstance(r["prefix"], str) else r["prefix"]))
    print(f"Gold (days): {gold_days:.2f}")
    print(f"Pred (days): {pred_days:.2f}")
    print("-"*60)
    
    table.add_data(
        r["case_id"], 
        int(r["k"]), 
        r["prefix"], 
        gold_days, 
        pred_days, 
        abs(gold_days - pred_days)
    )
wandb.log({"samples": table})

# %% Save model + artifacts
save_dir = f"/tmp/tabpfn_NT_{ts}"
os.makedirs(save_dir, exist_ok=True)

joblib.dump(model, os.path.join(save_dir, "tabpfn_nt_model.pkl"))
joblib.dump(x_word_dict, os.path.join(save_dir, "x_word_dict.pkl"))
with open(os.path.join(save_dir, "metadata.json"), "w") as f:
    json.dump({
        "seed": int(config["seed"]),
        "device": str(DEVICE),
        "maxlen": int(max_case_length),
        "vocab_size": int(vocab_size),
        "train_downsampled_to": int(min(len(X_train), config["sample_size"])) if config["sample_size"] else int(len(X_train))
    }, f)

# %% Save weights as artifact
artifact = wandb.Artifact(name=f"tabpfn_NT_artifacts_{config['dataset']}_{ts}", type="model")
artifact.add_dir(save_dir)
run.log_artifact(artifact)

# %%
wandb.finish()