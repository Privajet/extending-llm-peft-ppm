# %% TabPFN — Remaining Time (RT) prediction on BPI_Challenge_2012C
# - Temporal split by case start (aligned with your other scripts)
# - For each prefix of length k (k=1..n-1), target is RT_k = t_end - t_{k-1} in days
# - Features = fixed-length padded sequence of activities (OrdinalEncoder with PAD=0 conceptually)
# - Primary: TabPFNRegressor (if available). Fallback: TabPFNClassifier on quantile-binned targets
#   → converts class probs to days via expected value over bin representatives.
# - W&B logging + per-k MAE/MSE/RMSE curves, scatter, samples, artifact

import os, sys, random, logging, json, joblib
os.environ["MPLBACKEND"] = "Agg"  # headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
import numpy as np
import pandas as pd
import torch
import wandb

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Try TabPFN imports (prefer regressor)
regressor_ok = True
try:
    from tabpfn import TabPFNRegressor
except Exception:
    regressor_ok = False
from tabpfn import TabPFNClassifier

# %% Repro/Logging
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
if torch.cuda.is_available():
    log.info("GPU: %s", torch.cuda.get_device_name(0))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# %% Weights & Biases
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

CFG = dict(
    device=str(DEVICE),
    seed=SEED,
    mode=("regression" if regressor_ok else "classification_fallback"),
    # fallback-only params:
    n_bins=30,                # quantile bins for fallback classifier
    min_bins=8,
    max_bins=60
)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_tabpfn_RT_BPI_Challenge_2012C",
    entity="privajet-university-of-mannheim",
    name=f"tabpfn_RT_{ts}",
    config=CFG,
    resume="never",
    force=True
)

# %% Data
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_bpi_challenge.csv.gz")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)

# Build RT prefixes:
# For each case with events at times t_0..t_{n-1}, end_time = t_{n-1}
# For i in 1..n-1: prefix = acts[:i]; remaining_time (days) = (t_end - t_{i-1})
rows = []
for case_id, g in df.groupby("case:concept:name", sort=False):
    acts  = g["concept:name"].tolist()
    times = g["time:timestamp"].tolist()
    end_t = times[-1]
    for i in range(1, len(acts)):
        rows.append({
            "case_id": case_id,
            "prefix": acts[:i],
            "remaining_time": (end_t - times[i-1]).total_seconds() /  86400.0,  # days
            "k": i
        })
rt_df = pd.DataFrame(rows)
run.summary["total_prefixes"] = len(rt_df)

# Temporal split by case start
case_start = (
    df.groupby("case:concept:name")["time:timestamp"]
      .min().reset_index().sort_values("time:timestamp")
)
case_ids_sorted = case_start["case:concept:name"].tolist()
valid_ids = set(rt_df["case_id"])
case_ids = [cid for cid in case_ids_sorted if cid in valid_ids]

n_total = len(case_ids)
n_train = int(0.8 * n_total)
n_val   = int(0.2 * n_train)

train_ids = case_ids[: n_train - n_val]
val_ids   = case_ids[n_train - n_val : n_train]
test_ids  = case_ids[n_train : ]

train_df = rt_df[rt_df["case_id"].isin(train_ids)].reset_index(drop=True)
val_df   = rt_df[rt_df["case_id"].isin(val_ids)  ].reset_index(drop=True)
test_df  = rt_df[rt_df["case_id"].isin(test_ids) ].reset_index(drop=True)

wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# %% Feature encoding (OrdinalEncoder with a fixed PAD token first)
ALL_ACTS = sorted(df["concept:name"].unique().tolist())
PAD_TOKEN = "<PAD>"
CATS = [PAD_TOKEN] + ALL_ACTS

max_k = max(train_df["k"].max(), val_df["k"].max(), test_df["k"].max()) if len(rt_df) else 0
cols  = [f"e{i+1}" for i in range(max_k)]

def pad_prefix(p):
    return p + [PAD_TOKEN] * (max_k - len(p))

X_train_df = pd.DataFrame(train_df["prefix"].apply(pad_prefix).tolist(), columns=cols) if len(train_df) else pd.DataFrame(columns=cols)
X_val_df   = pd.DataFrame(val_df["prefix"].apply(pad_prefix).tolist(),   columns=cols) if len(val_df)   else pd.DataFrame(columns=cols)
X_test_df  = pd.DataFrame(test_df["prefix"].apply(pad_prefix).tolist(),  columns=cols) if len(test_df)  else pd.DataFrame(columns=cols)

# Stable categories per column (PAD=0, acts=1..). Map unknowns to PAD (0)
feat_enc = OrdinalEncoder(categories=[CATS]*max_k, handle_unknown="use_encoded_value", unknown_value=0.0) if max_k>0 else None
if max_k > 0:
    feat_enc.fit(pd.DataFrame([[PAD_TOKEN]*max_k], columns=cols))

X_train = feat_enc.transform(X_train_df).astype("float32") if max_k>0 else np.zeros((0,0), dtype="float32")
X_val   = feat_enc.transform(X_val_df).astype("float32")   if max_k>0 else np.zeros((0,0), dtype="float32")
X_test  = feat_enc.transform(X_test_df).astype("float32")  if max_k>0 else np.zeros((0,0), dtype="float32")

y_train_days = train_df["remaining_time"].astype("float32").to_numpy()
y_val_days   = val_df["remaining_time"].astype("float32").to_numpy()
y_test_days  = test_df["remaining_time"].astype("float32").to_numpy()

wandb.config.update({"max_k": int(max_k), "n_features": int(X_train.shape[1]) if len(X_train.shape) else 0}, allow_val_change=True)

# %% Train TabPFN model
model = None
metadata = {"mode": wandb.config["mode"]}

if regressor_ok:
    log.info("Initializing TabPFNRegressor for RT.")
    model = TabPFNRegressor(device=str(DEVICE), seed=SEED)
    model.fit(X_train, y_train_days)
else:
    log.warning("TabPFNRegressor not available. Falling back to TabPFNClassifier on quantile-binned RT.")
    # Choose number of bins within [min_bins, max_bins], capped by unique values
    n_bins = int(np.clip(CFG["n_bins"], CFG["min_bins"], CFG["max_bins"]))
    n_bins = int(min(n_bins, max(2, len(np.unique(y_train_days)))))  # ensure >=2 bins, <= unique y

    # Quantile binning on train; drop duplicate edges if necessary
    qcuts = pd.qcut(y_train_days, q=n_bins, duplicates="drop")
    bin_intervals = qcuts.cat.categories
    bin_codes_train = qcuts.cat.codes.to_numpy()

    # Representative days per bin (mean of train y in bin; mid if empty)
    reps = []
    for c in range(len(bin_intervals)):
        mask = (bin_codes_train == c)
        if mask.any():
            reps.append(float(y_train_days[mask].mean()))
        else:
            iv = bin_intervals[c]
            reps.append(float((float(iv.left) + float(iv.right)) / 2.0))
    reps = np.array(reps, dtype="float32")

    # Build edges (right-closed) to encode val/test consistently
    edges = np.unique(np.concatenate(([bin_intervals[0].left], [iv.right for iv in bin_intervals])))
    edges = np.array(edges, dtype="float64")
    eps = 1e-8
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + eps

    def to_bin_codes(y):
        idx = np.clip(np.digitize(y, edges[1:-1], right=True), 0, len(edges)-2)
        return idx.astype("int64")

    y_train_cls = to_bin_codes(y_train_days)
    y_val_cls   = to_bin_codes(y_val_days)
    y_test_cls  = to_bin_codes(y_test_days)

    clf = TabPFNClassifier(device=str(DEVICE), seed=SEED)
    clf.fit(X_train, y_train_cls)
    model = clf

    metadata.update({
        "fallback_bins": int(len(edges)-1),
        "bin_edges": edges.tolist()
    })

# %% Prediction helper
def predict_days(X):
    """Predict days (RT) for TabPFNRegressor or classifier fallback."""
    if regressor_ok:
        y_hat = model.predict(X)  # (N,)
        return y_hat.astype("float32")
    else:
        # Expected value over bin representatives if proba available
        try:
            proba = model.predict_proba(X)  # (N, B)
            B = proba.shape[1]
            reps_vec = reps[:B] if len(reps) >= B else np.pad(reps, (0, B - len(reps)), constant_values=reps[-1])
            y_exp = (proba * reps_vec[None, :]).sum(axis=1)
            return y_exp.astype("float32")
        except Exception:
            y_cls = model.predict(X).astype("int64")
            reps_vec = reps
            y_hat = np.array([reps_vec[i] if i < len(reps_vec) else reps_vec[-1] for i in y_cls], dtype="float32")
            return y_hat

# %% Validation quick check
y_val_pred = predict_days(X_val)
val_mae = float(mean_absolute_error(y_val_days, y_val_pred)) if len(y_val_days) else float("nan")
val_mse = float(mean_squared_error(y_val_days, y_val_pred))  if len(y_val_days) else float("nan")
val_rmse = float(np.sqrt(val_mse)) if np.isfinite(val_mse) else float("nan")
wandb.log({"val/mae_days": val_mae, "val/mse_days": val_mse, "val/rmse_days": val_rmse})

# %% Test per-k eval
y_test_pred = predict_days(X_test)

k_vals, counts = [], []
maes, mses, rmses = [], [], []

max_k = int(test_df["k"].max()) if len(test_df) else 0
for k in range(1, max_k + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue
    idx = subset.index.values
    yt = y_test_days[idx]
    yp = y_test_pred[idx]

    mae  = mean_absolute_error(yt, yp)
    mse  = mean_squared_error(yt, yp)
    rmse = float(np.sqrt(mse))

    k_vals.append(k); counts.append(len(idx))
    maes.append(float(mae)); mses.append(float(mse)); rmses.append(float(rmse))

avg_mae  = float(np.mean(maes))  if maes  else float("nan")
avg_mse  = float(np.mean(mses))  if mses  else float("nan")
avg_rmse = float(np.mean(rmses)) if rmses else float("nan")

print(f"Average MAE across all prefixes:  {avg_mae:.2f} days")
print(f"Average MSE across all prefixes:  {avg_mse:.2f} (days^2)")
print(f"Average RMSE across all prefixes: {avg_rmse:.2f} days")

# %% Plots → disk
plot_dir = "/ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/plots/Baselines/TabPFN/RT"
os.makedirs(plot_dir, exist_ok=True)

if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, maes, marker="o", label="MAE")
    plt.title("RT (TabPFN) — MAE vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("MAE (days)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mae_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, rmses, marker="o", label="RMSE")
    plt.title("RT (TabPFN) — RMSE vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("RMSE (days)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"rmse_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, mses, marker="o", label="MSE")
    plt.title("RT (TabPFN) — MSE vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("MSE (days^2)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mse_vs_k_{ts}.png"), dpi=150); plt.close()

print(f"Saved plots to: {plot_dir}")

# %% Log curves + macro to W&B
wandb.log({
    "curves/k": k_vals,
    "curves/counts": counts,
    "curves/mae": maes,
    "curves/mse": mses,
    "curves/rmse": rmses,
    "metrics/avg_mae":  avg_mae,
    "metrics/avg_mse":  avg_mse,
    "metrics/avg_rmse": avg_rmse,
    "metadata": metadata
})

# %% Global scatter / hist (days)
if len(y_test_days):
    abs_err = np.abs(y_test_days - y_test_pred)
    tab = wandb.Table(
        data=[[float(y_test_days[i]), float(y_test_pred[i]), float(abs_err[i])] for i in range(len(abs_err))],
        columns=["true_days", "pred_days", "abs_err_days"]
    )
    wandb.log({
        "scatter_true_vs_pred": wandb.plot.scatter(tab, "true_days", "pred_days", title="RT (TabPFN): True vs Pred (days)"),
        "error_hist": wandb.Histogram(abs_err)
    })

# %% Inference helper (days)
def predict_remaining_days(prefix_tokens):
    x = pd.DataFrame([pad_prefix(prefix_tokens)], columns=cols)
    x_enc = feat_enc.transform(x).astype("float32")
    return float(predict_days(x_enc)[0])

# %% Sample predictions (print + W&B table)
sample = test_df.sample(n=min(5, len(test_df)), random_state=42) if len(test_df) else test_df
s_table = wandb.Table(columns=["case_id","k","prefix","gold_days","pred_days","abs_err_days"])
for _, r in sample.iterrows():
    pred = float(predict_remaining_days(r["prefix"]))
    gold = float(r["remaining_time"])
    s_table.add_data(r["case_id"], r["k"], " → ".join(r["prefix"]), gold, pred, abs(gold - pred))
    print("Prefix:", " → ".join(r["prefix"]))
    print(f"Gold (days): {gold:.2f} | Pred (days): {pred:.2f} | Abs err: {abs(gold - pred):.2f}")
    print("-"*60)
wandb.log({"samples": s_table})

# %% Save model + artifacts
save_dir = f"/tmp/tabpfn_RT_{ts}"
os.makedirs(save_dir, exist_ok=True)

# Persist model
joblib.dump(model, os.path.join(save_dir, "tabpfn_rt_model.pkl"))

# Persist encoders + metadata needed for inference
with open(os.path.join(save_dir, "feature_columns.json"), "w") as f:
    json.dump({"cols": cols, "categories": CATS}, f)
joblib.dump(feat_enc, os.path.join(save_dir, "ordinal_encoder.pkl"))

meta_to_save = {
    "mode": wandb.config["mode"],
    "seed": SEED,
    "device": str(DEVICE),
    "max_k": int(max_k)
}
if not regressor_ok:
    meta_to_save.update({
        "bin_edges": metadata.get("bin_edges", []),
        "n_bins": metadata.get("fallback_bins", 0)
    })
with open(os.path.join(save_dir, "metadata.json"), "w") as f:
    json.dump(meta_to_save, f)

artifact = wandb.Artifact(name=f"tabpfn_RT_artifacts_{ts}", type="model")
artifact.add_dir(save_dir)
run.log_artifact(artifact)

# %%
wandb.finish()