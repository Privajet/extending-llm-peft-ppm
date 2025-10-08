# %% LSTM baseline — Next-Time (NT) prediction on HelpDesk

import os
os.environ["MPLBACKEND"] = "Agg"  # headless matplotlib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
import wandb
from wandb.integration.keras import WandbMetricsLogger

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from sklearn import metrics

# %% W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %% Config
config = {
    "checkpoint_path": "/tmp/best_lstm_nt_HelpDesk.weights.h5",
    "monitor_metric": "val_loss",
    "monitor_mode": "min",
    "pad_direction": "pre",
    "truncating": "pre",
    "early_stop_patience":  7,
    "reduce_lr_factor":     0.5,
    "reduce_lr_patience":   3,
    "min_lr":               1e-6,
    "max_ctx":              30,
    "weight_decay":         1e-4,
    "learning_rate":        3e-4,       # 5e-4 → 3e-4 (20.09.)
    "batch_size":           64,         # 32 → 64 (20.09.)
    "epochs":               90,
    "embedding_dim":        256,        # 64 → 256 (20.09.)
    "lstm_units_1":         256,        # 128 → 256 (20.09.)
    "lstm_units_2":         128,        # 64 → 128 (20.09.)
    "dropout":              0.30,       # 0.20 → 0.30 (20.09.)
    "recurrent_dropout":    0.10,       # added (20.09.)
    "l2":                   1e-5,      
    "clipnorm":             1.0,
    "use_huber":            True,       # added (20.09.)
    "huber_delta":          0.5,        # added (20.09.)
    "scale_range":          (-1, 1)
}

# %% Init W&B
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_lstm_NT_HelpDesk",
    entity="privajet-university-of-mannheim",
    name=f"lstm_nt_{ts}",
    config=config
)

CHECKPOINT_PATH = config["checkpoint_path"]
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

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
# %%
PROC_DIR   = "/ceph/lfertig/Thesis/data/HelpDesk/processed"
META_PATH  = os.path.join(PROC_DIR, "metadata.json")

with open(META_PATH, "r") as f:
    meta = json.load(f)

x_word_dict = meta["x_word_dict"]  # includes [PAD]=0, [UNK]=1
vocab_size  = len(x_word_dict)
PAD_ID = x_word_dict["[PAD]"]      # should be 0
UNK_ID = x_word_dict["[UNK]"]      # should be 1

# Use real prefix lengths, capped by max_ctx
maxlen = min(config["max_ctx"], max(len(p) for p in train_df["prefix"]))

def encode_prefix(tokens):
    # tokens is a list like ["assign-seriousness", "take-in-charge-ticket", ...]
    return [x_word_dict.get(t, UNK_ID) for t in tokens]

def prepare_data(frame):
    X = [encode_prefix(p) for p in frame["prefix"]]
    X = pad_sequences(
        X, maxlen=maxlen,
        padding=config["pad_direction"], truncating=config["truncating"], value=PAD_ID
    ).astype("int32")
    # log1p to keep positivity and stabilize
    y_log = np.log1p(frame["next_time_delta"].astype("float32").to_numpy()).reshape(-1, 1)
    return X, y_log
    
X_train, y_train = prepare_data(train_df)
X_val,   y_val   = prepare_data(val_df)
X_test,  y_test  = prepare_data(test_df)

def encode_seq(tokens): 
    return encode_prefix(tokens)

# %% Model
layers = [
    Embedding(
        input_dim=vocab_size,  # vocab_size already includes PAD(0) and UNK(1)
        output_dim=config["embedding_dim"],
        mask_zero=True
    ),
    LSTM(
        config["lstm_units_1"],
        return_sequences=(config["lstm_units_2"] > 0),
        dropout=config["dropout"],
        recurrent_dropout=config["recurrent_dropout"],
        kernel_regularizer=l2(config["l2"]),
        recurrent_initializer="orthogonal",
        recurrent_regularizer=l2(config["l2"])
    ),
    Dropout(config["dropout"])
]

if config["lstm_units_2"] > 0:
    layers += [
        LSTM(
            config["lstm_units_2"],
            dropout=config["dropout"],
            recurrent_dropout=config["recurrent_dropout"],
            kernel_regularizer=l2(config["l2"]),
            recurrent_initializer="orthogonal",
            recurrent_regularizer=l2(config["l2"])
        ),
        Dropout(config["dropout"])
    ]

layers += [Dense(64, activation="relu", kernel_regularizer=l2(config["l2"])), 
           Dropout(config["dropout"]), 
           Dense(1, activation="linear")]
model = Sequential(layers)

optimizer = AdamW(learning_rate=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                    clipnorm=config["clipnorm"])

loss = (tf.keras.losses.Huber(delta=config["huber_delta"]) if config["use_huber"] else "mse")

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
)

# %% Callbacks
checkpoint_cb = ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    save_weights_only=True,
    monitor=config["monitor_metric"],
    save_best_only=True,
    mode=config["monitor_mode"],
    verbose=1
)
early_stop = EarlyStopping(
    monitor=config["monitor_metric"],
    patience=config["early_stop_patience"],
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor=config["monitor_metric"],
    factor=config["reduce_lr_factor"],
    patience=config["reduce_lr_patience"],
    min_lr=config["min_lr"],
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    callbacks=[checkpoint_cb, early_stop, reduce_lr, WandbMetricsLogger()],
    verbose=2
)

# Safety: ensure checkpoint exists then load best
if not os.path.exists(CHECKPOINT_PATH):
    model.save_weights(CHECKPOINT_PATH)
model.load_weights(CHECKPOINT_PATH)

# %% Inference helper (returns days)
def predict_delta_days(prefix_tokens):
    x = pad_sequences([encode_seq(prefix_tokens)], maxlen=maxlen,
                      padding=config["pad_direction"], truncating=config["truncating"], value=PAD_ID)
    y_log = model.predict(x, verbose=0)[0, 0]
    y_days = float(np.expm1(y_log))
    return max(0.0, y_days)

# %% Per-k evaluation (metrics in days)
k_vals, counts, maes, mses, rmses = [], [], [], [], []
for k in sorted(test_df["k"].astype(int).unique()):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue
    X_t = pad_sequences([encode_seq(p) for p in subset["prefix"]], maxlen=maxlen,
                    padding=config["pad_direction"], truncating=config["truncating"], value=PAD_ID)
    y_true = subset["next_time_delta"].values.reshape(-1, 1).astype("float32")
    y_true = np.maximum(y_true, 0.0)

    # predict (log) then inverse-transform
    y_pred_log = model.predict(X_t, verbose=0)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0.0)

    k_vals.append(k); counts.append(len(subset))
    mae  = metrics.mean_absolute_error(y_true, y_pred)
    mse  = metrics.mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    maes.append(mae); mses.append(mse); rmses.append(rmse)

# Macro averages across k-bins
avg_mae  = float(np.mean(maes))  if maes  else float("nan")
avg_mse  = float(np.mean(mses))  if mses  else float("nan")
avg_rmse = float(np.mean(rmses)) if rmses else float("nan")

print(f"Average MAE across all prefixes:  {avg_mae:.2f} days")
print(f"Average MSE across all prefixes:  {avg_mse:.2f} (days^2)")
print(f"Average RMSE across all prefixes: {avg_rmse:.2f} days")

# %% Plots → disk
plot_dir = "/ceph/lfertig/Thesis/notebook/HelpDesk/plots/Baselines/LSTM/NT"
os.makedirs(plot_dir, exist_ok=True)

h = history.history

# Training curves (log-space targets)
plt.figure(figsize=(8,5))
plt.plot(h["loss"], label="Train Loss")
plt.plot(h["val_loss"], label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch"); plt.ylabel("Huber Loss (log-space)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"loss_{ts}.png"), dpi=150); plt.close()

plt.figure(figsize=(8,5))
plt.plot(h["mae"], label="Train MAE (log-space)")
plt.plot(h["val_mae"], label="Val MAE (log-space)")
plt.title("MAE over Epochs (log-space)")
plt.xlabel("Epoch"); plt.ylabel("MAE (log-space)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"mae_log_{ts}.png"), dpi=150); plt.close()

# Per-k (days)
if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, maes, marker='o', label='MAE (days)')
    plt.title('MAE vs. Prefix Length (k)'); plt.xlabel('Prefix Length (k)'); plt.ylabel('MAE (days)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mae_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, rmses, marker='o', label='RMSE (days)')
    plt.title('RMSE vs. Prefix Length (k)'); plt.xlabel('Prefix Length (k)'); plt.ylabel('RMSE (days)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"rmse_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, mses, marker='o', label='MSE (days^2)')
    plt.title('MSE vs. Prefix Length (k)'); plt.xlabel('Prefix Length (k)'); plt.ylabel('MSE')
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
X_all = pad_sequences([encode_seq(p) for p in test_df["prefix"]], maxlen=maxlen,
                      padding=config["pad_direction"], truncating=config["truncating"], value=PAD_ID)
y_true_all = test_df["next_time_delta"].values.reshape(-1, 1)
y_pred_all = np.expm1(model.predict(X_all, verbose=0))
y_pred_all = np.maximum(y_pred_all, 0.0)
abs_err = np.abs(y_true_all - y_pred_all).reshape(-1)

tab = wandb.Table(
    data=[[float(y_true_all[i,0]), float(y_pred_all[i,0]), float(abs_err[i])] for i in range(len(abs_err))],
    columns=["true_days", "pred_days", "abs_err_days"]
)
wandb.log({
    "scatter_true_vs_pred": wandb.plot.scatter(tab, "true_days", "pred_days", title="NT: True vs Pred (days)"),
    "error_hist": wandb.Histogram(abs_err),
})

# %% Sample predictions (days)
sample = test_df.sample(n=min(5, len(test_df)), random_state=42) if len(test_df) else test_df
s_table = wandb.Table(columns=["case_id","k","prefix","gold_days","pred_days","abs_err_days"])
for _, r in sample.iterrows():
    pred = predict_delta_days(r["prefix"])
    gold = float(r["next_time_delta"])
    s_table.add_data(r["case_id"], r["k"], " → ".join(r["prefix"]), gold, pred, abs(gold - pred))
    print("Prefix:", " → ".join(r["prefix"]))
    print(f"Gold (days): {gold:.2f}")
    print(f"Pred (days): {pred:.2f}")
    print("-"*60)
wandb.log({"samples": s_table})

# %% Save checkpoint as W&B artifact
artifact = wandb.Artifact("lstm_nt_model_helpdesk", type="model")
artifact.add_file(CHECKPOINT_PATH)
run.log_artifact(artifact)

# %%
wandb.finish()