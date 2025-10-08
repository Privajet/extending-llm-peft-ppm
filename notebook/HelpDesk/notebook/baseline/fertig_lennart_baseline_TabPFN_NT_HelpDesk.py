# %% TabPFN — Next-Time (NT) prediction on HelpDesk
# - Temporal split by case start (aligned with your other scripts)
# - For each prefix of length k (k=1..n-1), target is Δt_k = t_k - t_{k-1} in days
# - Features = fixed-length padded sequence of activity IDS from metadata (PAD=0, UNK=1), using pre-trunc + pre-pad
# - Primary: TabPFNRegressor (if available). Fallback: TabPFNClassifier on quantile-binned LOG targets
#   → converts class probs to LOG-days via expected value, then expm1 back to days
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

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Try TabPFN imports (prefer regressor)
regressor_ok = True
try:
    from tabpfn import TabPFNRegressor
except Exception:
    regressor_ok = False

from tabpfn import TabPFNClassifier
try:
    from tabpfn_extensions.many_class import ManyClassClassifier
except Exception:
    ManyClassClassifier = None

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

config = dict(
    device=str(DEVICE),
    seed=SEED,
    mode=("regression" if regressor_ok else "classification_fallback"),
    # fallback-only params:
    n_bins=30,                # quantile bins for fallback classifier
    min_bins=8,
    max_bins=60,
    # TabPFN
    sample_size=10000,        # downsample train for speed/stability; set None to disable
)
config["max_ctx"] = 30

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_tabpfn_NT_HelpDesk",
    entity="privajet-university-of-mannheim",
    name=f"tabpfn_NT_{ts}",
    config=config,
    resume="never",
    force=True
)

# %% Data
train_df = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_time_train.csv")
val_df   = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_time_val.csv")
test_df  = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_time_test.csv")

for d in (train_df, val_df, test_df):
    d["prefix"] = d["prefix"].astype(str).str.split()
    # NT target standardization
    if "next_time_delta" not in d.columns:
            d.rename(columns={"next_time": "next_time_delta"}, inplace=True)

print(f"Train prefixes: {len(train_df)} - Validation prefixes: {len(val_df)} - Test prefixes: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# %% Feature encoding        
PROC_DIR  = "/ceph/lfertig/Thesis/data/HelpDesk/processed"
META_PATH = os.path.join(PROC_DIR, "metadata.json")
with open(META_PATH, "r") as f:
    meta = json.load(f)

x_word_dict = meta["x_word_dict"]  # {"[PAD]":0, "[UNK]":1, ...}
PAD_ID = x_word_dict["[PAD]"]; UNK_ID = x_word_dict["[UNK]"]
vocab_size = len(x_word_dict)

MAX_CTX = config["max_ctx"]
maxlen = min(MAX_CTX, max(len(p) for p in train_df["prefix"])) if len(train_df) else MAX_CTX

def encode_prefix(tokens):
    return [x_word_dict.get(t, UNK_ID) for t in tokens]

def prepare_X(frame):
    X = [encode_prefix(p) for p in frame["prefix"]]
    X = pad_sequences(
        X, maxlen=maxlen,
        padding="pre", truncating="pre", value=PAD_ID
    ).astype("float32")
    return X

def prepare_y_log1p_days(frame):
    y_days = frame["next_time_delta"].astype("float32").to_numpy()
    y_days = np.maximum(y_days, 0.0)   # safety clamp
    y_log  = np.log1p(y_days)
    return y_log

# Build matrices
X_train = prepare_X(train_df)
X_val   = prepare_X(val_df)
X_test  = prepare_X(test_df)

# Targets (log-space for training; days for reporting)
y_train_log = prepare_y_log1p_days(train_df)
y_val_log   = prepare_y_log1p_days(val_df)
y_test_log  = prepare_y_log1p_days(test_df)   # diagnostics only

y_train_days = train_df["next_time_delta"].astype("float32").to_numpy()
y_val_days   = val_df["next_time_delta"].astype("float32").to_numpy()
y_test_days  = test_df["next_time_delta"].astype("float32").to_numpy()

wandb.config.update({
    "maxlen": int(maxlen),
    "vocab_size": int(vocab_size),
    "n_features": int(X_train.shape[1])
}, allow_val_change=True)

# %% Train TabPFN model
metadata = {}
SAMPLE_SIZE = config.get("sample_size", 10000)

def _downsample(X, y, seed):
    if SAMPLE_SIZE and len(X) > SAMPLE_SIZE:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(X), size=SAMPLE_SIZE, replace=False)
        X_fit = X[sel]
        y_fit = y[sel]
        log.info("Downsampled train from %d to %d for TabPFN", len(X), len(X_fit))
        wandb.log({"train_downsampled_to": int(len(X_fit))})
        return X_fit, y_fit
    return X, y

if regressor_ok:
    # Primary: Regressor in log-space
    log.info("Initializing TabPFNRegressor for NT (log-space).")
    model = TabPFNRegressor(device=str(DEVICE), ignore_pretraining_limits=True)
    for key in ("inference_batch_size", "inference_max_batch_size", "batch_size_inference"):
        try:
            model.set_params(**{key: 256}); break
        except Exception:
            pass
    X_train_fit, y_train_fit = _downsample(X_train, y_train_log, SEED)
    model.fit(X_train_fit, y_train_fit)
    metadata["mode"] = "regression"

else:
    # Fallback: TabPFNClassifier on quantile-binned LOG targets; proba → E[y_log] → expm1
    log.warning("TabPFNRegressor not available. Fallback: TabPFNClassifier on quantile-binned log targets.")
    n_bins = int(np.clip(config["n_bins"], config["min_bins"], config["max_bins"]))
    n_bins = int(min(n_bins, max(2, len(np.unique(y_train_log)))))

    qcuts = pd.qcut(y_train_log, q=n_bins, duplicates="drop")
    bin_intervals = qcuts.cat.categories
    bin_codes_train = qcuts.cat.codes.to_numpy()

    # bin representatives in log-space
    reps = []
    for c in range(len(bin_intervals)):
        mask = (bin_codes_train == c)
        if mask.any():
            reps.append(float(y_train_log[mask].mean()))
        else:
            iv = bin_intervals[c]
            reps.append(float((float(iv.left) + float(iv.right)) / 2.0))
    reps = np.array(reps, dtype="float32")

    # right-closed edges in log-space for consistent val/test binning
    edges = np.unique(np.concatenate(([bin_intervals[0].left], [iv.right for iv in bin_intervals]))).astype("float64")
    eps = 1e-8
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + eps

    def to_bin_codes_log(y_log):
        idx = np.clip(np.digitize(y_log, edges[1:-1], right=True), 0, len(edges)-2)
        return idx.astype("int64")

    y_train_cls = to_bin_codes_log(y_train_log)
    y_val_cls   = to_bin_codes_log(y_val_log)
    y_test_cls  = to_bin_codes_log(y_test_log)

    # ACT-style ManyClass wrapper (use #bins as "classes")
    n_classes = int(len(np.unique(y_train_cls)))
    use_manyclass = (ManyClassClassifier is not None) and (n_classes > 10)
    if use_manyclass:
        log.info("Using ManyClassClassifier wrapper for >10 classes.")
    else:
        log.warning("ManyClassClassifier unavailable or not needed (<=10 classes).")

    base_clf = TabPFNClassifier(device=str(DEVICE), ignore_pretraining_limits=True)
    for key in ("inference_batch_size", "inference_max_batch_size", "batch_size_inference"):
        try:
            base_clf.set_params(**{key: 256}); break
        except Exception:
            pass

    clf = (ManyClassClassifier(
              estimator=base_clf,
              alphabet_size=min(10, n_classes),
              n_estimators_redundancy=3,
              random_state=SEED,
              verbose=1
           ) if use_manyclass else base_clf)

    X_train_fit, y_train_fit = _downsample(X_train, y_train_cls, SEED)
    clf.fit(X_train_fit, y_train_fit)
    model = clf
    metadata.update({
        "mode": "classification_fallback",
        "fallback_bins": int(len(edges)-1),
        "bin_edges_log": edges.tolist()
    })

# (optional) record which mode actually used
wandb.config.update({"mode_used": metadata.get("mode", "unknown")}, allow_val_change=True)

# %% Prediction helper
def predict_days(X):
    if metadata.get("mode") == "regression":
        y_hat_log = model.predict(X).astype("float32")
        y_days = np.expm1(y_hat_log)
        return np.maximum(y_days, 0.0)
    # classification fallback → E[y_log] then expm1
    try:
        proba = model.predict_proba(X)  # (N, B)
        B = proba.shape[1]
        reps_vec = reps[:B] if len(reps) >= B else np.pad(reps, (0, B - len(reps)), constant_values=reps[-1])
        y_exp_log = (proba * reps_vec[None, :]).sum(axis=1)
        y_days = np.expm1(y_exp_log)
        return np.maximum(y_days.astype("float32"), 0.0)
    except Exception:
        y_cls = model.predict(X).astype("int64")
        reps_vec = reps
        y_log = np.array([reps_vec[i] if i < len(reps_vec) else reps_vec[-1] for i in y_cls], dtype="float32")
        y_days = np.expm1(y_log)
        return np.maximum(y_days, 0.0)

# %% Validation quick check (metrics in days)
y_val_pred = predict_days(X_val)
val_mae = float(mean_absolute_error(y_val_days, y_val_pred)) if len(y_val_days) else float("nan")
val_mse = float(mean_squared_error(y_val_days, y_val_pred))  if len(y_val_days) else float("nan")
val_rmse = float(np.sqrt(val_mse)) if np.isfinite(val_mse) else float("nan")
wandb.log({"val/mae_days": val_mae, "val/mse_days": val_mse, "val/rmse_days": val_rmse})

# %% Test per-k eval (days)
y_test_pred = predict_days(X_test)

k_vals, counts = [], []
maes, mses, rmses = [], [], []

max_k = int(test_df["k"].max()) if "k" in test_df.columns and len(test_df) else maxlen
for k in sorted(test_df["k"].astype(int).unique() if "k" in test_df.columns else range(1, max_k+1)):
    subset = test_df[test_df["k"] == k] if "k" in test_df.columns else test_df
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

# %%  Plots
plot_dir = "/ceph/lfertig/Thesis/notebook/HelpDesk/plots/Baselines/TabPFN/NT"
os.makedirs(plot_dir, exist_ok=True)

if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, maes, marker="o", label="MAE")
    plt.title("NT (TabPFN) — MAE vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("MAE (days)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mae_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, rmses, marker="o", label="RMSE")
    plt.title("NT (TabPFN) — RMSE vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("RMSE (days)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"rmse_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, mses, marker="o", label="MSE")
    plt.title("NT (TabPFN) — MSE vs. Prefix Length (k)")
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
        "scatter_true_vs_pred": wandb.plot.scatter(tab, "true_days", "pred_days", title="NT (TabPFN): True vs Pred (days)"),
        "error_hist": wandb.Histogram(abs_err)
    })

# %% Inference helper (days)
def predict_delta_days(prefix_tokens):
    x = pad_sequences([encode_prefix(prefix_tokens)], maxlen=maxlen,
                      padding="pre", truncating="pre", value=PAD_ID).astype("float32")  # (1, L)
    y = predict_days(x)
    return float(np.maximum(y[0], 0.0))

# %% Sample predictions (print + W&B table)
sample = test_df.sample(n=min(5, len(test_df)), random_state=42) if len(test_df) else test_df
s_table = wandb.Table(columns=["case_id","k","prefix","gold_days","pred_days","abs_err_days"])
for _, r in sample.iterrows():
    pred = float(predict_delta_days(r["prefix"]))
    gold = float(r["next_time_delta"])
    s_table.add_data(r.get("case_id", ""), r.get("k", ""), " → ".join(r["prefix"]), gold, pred, abs(gold - pred))
    print("Prefix:", " → ".join(r["prefix"]))
    print(f"Gold (days): {gold:.2f} | Pred (days): {pred:.2f} | Abs err: {abs(gold - pred):.2f}")
    print("-"*60)
wandb.log({"samples": s_table})

# %% Save model + artifacts (same style as RT)
save_dir = f"/tmp/tabpfn_NT_{ts}"
os.makedirs(save_dir, exist_ok=True)

# Persist model
joblib.dump(x_word_dict, os.path.join(save_dir, "x_word_dict.pkl"))
with open(os.path.join(save_dir, "metadata.json"), "w") as f:
    json.dump({
        "mode": metadata.get("mode", "unknown"),
        "seed": int(SEED),
        "device": str(DEVICE),
        "maxlen": int(maxlen),
        "vocab_size": int(vocab_size),
        **({
            "bin_edges_log": metadata.get("bin_edges_log", []),
            "n_bins": metadata.get("fallback_bins", 0)
        } if metadata.get("mode") == "classification_fallback" else {})
    }, f)

artifact = wandb.Artifact(name=f"tabpfn_NT_artifacts_{ts}", type="model")
artifact.add_dir(save_dir)
run.log_artifact(artifact)

# %%
wandb.finish()