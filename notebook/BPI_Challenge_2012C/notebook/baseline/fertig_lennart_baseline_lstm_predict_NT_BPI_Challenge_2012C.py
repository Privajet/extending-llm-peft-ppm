# %% LSTM baseline — Next-Time (NT) prediction on BPI_Challenge_2012C

import os
os.environ["MPLBACKEND"] = "Agg"  # headless matplotlib

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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import metrics

# %% W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %% Config
config = {
    "learning_rate": 5e-4,
    "batch_size":    32,
    "epochs":        90,
    "embedding_dim": 64,
    "lstm_units_1":  128,
    "lstm_units_2":  64,     # set 0 to disable 2nd LSTM
    "dropout":       0.20,
    "l2":            1e-5,
    "clipnorm":      1.0,
    "use_huber":     False,   # True -> Huber(delta=0.25), False -> MSE
    "scale_range":   (-1, 1)
}

# %% Init W&B
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_lstm_NT_BPI_Challenge_2012C",
    entity="privajet-university-of-mannheim",
    name=f"lstm_nt_{ts}",
    config=config
)

CKPT_PATH = "/tmp/best_lstm_nt_BPI_Challenge_2012C.weights.h5"
os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

# %% Data
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_bpi_challenge.csv.gz")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)

# Build NT prefixes (align k with ACT code):
# for each case with events e_0..e_{n-1}, for i in 1..n-1
#   prefix = acts[:i]
#   next_time_delta (days) = t_i - t_{i-1}
rows = []
for case_id, g in df.groupby("case:concept:name", sort=False):
    acts  = g["concept:name"].tolist()
    times = g["time:timestamp"].tolist()
    for i in range(1, len(acts)):
        rows.append({
            "case_id": case_id,
            "prefix": acts[:i],
            "next_time_delta": (times[i] - times[i-1]).total_seconds() / 86400.0, #days
            "k": i
        })
nt_df = pd.DataFrame(rows)

# Temporal split by case start (as in ACT)
case_start = (
    df.groupby("case:concept:name")["time:timestamp"]
      .min().reset_index().sort_values("time:timestamp")
)
case_ids = case_start["case:concept:name"].tolist()

n_total = len(case_ids)
n_train = int(n_total * 0.8)
n_val   = int(n_train * 0.2)

train_ids = case_ids[:n_train - n_val]
val_ids   = case_ids[n_train - n_val:n_train]
test_ids  = case_ids[n_train:]

train_df = nt_df[nt_df["case_id"].isin(train_ids)].reset_index(drop=True)
val_df   = nt_df[nt_df["case_id"].isin(val_ids)].reset_index(drop=True)
test_df  = nt_df[nt_df["case_id"].isin(test_ids)].reset_index(drop=True)

print(f"Train prefixes: {len(train_df)} - Val: {len(val_df)} - Test: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# %% Encode activities + pad
le = LabelEncoder().fit(df["concept:name"])
PAD_ID = 0
def encode_seq(seq): return le.transform(seq) + 1  # keep 0 for PAD

maxlen = int(nt_df["k"].max()) if len(nt_df) else 0

# Scale y to [-1,1] for stability; invert for metrics
def prepare_data(frame, scaler=None, fit_scaler=True):
    X = [encode_seq(p) for p in frame["prefix"]]
    X = pad_sequences(X, maxlen=maxlen, padding="post", value=PAD_ID)  # right-pad

    y = frame["next_time_delta"].values.reshape(-1, 1)

    if fit_scaler:
        scaler = MinMaxScaler(feature_range=config["scale_range"])
        y_scaled = scaler.fit_transform(y)
        return X, y_scaled, scaler
    else:
        y_scaled = scaler.transform(y)
        return X, y_scaled

X_train, y_train, scaler = prepare_data(train_df, fit_scaler=True)
X_val,   y_val           = prepare_data(val_df,   scaler=scaler, fit_scaler=False)
X_test,  y_test          = prepare_data(test_df,  scaler=scaler, fit_scaler=False)

# %% Model
layers = [
    Embedding(
        input_dim=len(le.classes_) + 1,  # +1 for PAD
        output_dim=config["embedding_dim"],
        input_length=maxlen,
        mask_zero=True
    ),
    LSTM(
        config["lstm_units_1"],
        return_sequences=(config["lstm_units_2"] > 0),
        dropout=config["dropout"],
        recurrent_dropout=0.0,
        kernel_regularizer=l2(config["l2"]),
        recurrent_regularizer=l2(config["l2"])
    ),
    Dropout(config["dropout"])
]

if config["lstm_units_2"] > 0:
    layers += [
        LSTM(
            config["lstm_units_2"],
            dropout=config["dropout"],
            recurrent_dropout=0.0,
            kernel_regularizer=l2(config["l2"]),
            recurrent_regularizer=l2(config["l2"])
        ),
        Dropout(config["dropout"])
    ]

layers += [Dense(1, activation="linear")]
model = Sequential(layers)

optimizer = Adam(learning_rate=config["learning_rate"], clipnorm=config["clipnorm"])
loss = (tf.keras.losses.Huber(delta=0.25) if config["use_huber"] else "mse")

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
)

# %% Callbacks (monitor val_mae on scaled targets — good proxy; true days reported below)
checkpoint_cb = ModelCheckpoint(
    filepath=CKPT_PATH,
    save_weights_only=True,
    monitor="val_mae",
    save_best_only=True,
    mode="min",
    verbose=1
)
early_stop = EarlyStopping(
    monitor="val_mae",
    patience=7,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_mae",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
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
if not os.path.exists(CKPT_PATH):
    model.save_weights(CKPT_PATH)
model.load_weights(CKPT_PATH)

# %% Inference helper (returns days)
def predict_delta_days(prefix_tokens):
    x = pad_sequences([encode_seq(prefix_tokens)], maxlen=maxlen, padding="post", value=PAD_ID)
    y_scaled = model.predict(x, verbose=0)[0, 0]
    return float(scaler.inverse_transform([[y_scaled]])[0, 0])

# %% Per-k evaluation (metrics in days)
k_vals, counts = [], []
maes, mses, rmses = [], [], []

for k in range(1, maxlen + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue

    X_t = pad_sequences([encode_seq(p) for p in subset["prefix"]], maxlen=maxlen, padding="post", value=PAD_ID)
    y_true = subset["next_time_delta"].values.reshape(-1, 1)
    # predict (scaled) then inverse-transform
    y_pred_scaled = model.predict(X_t, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)

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
plot_dir = "/ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/plots/Baselines/LSTM/NT"
os.makedirs(plot_dir, exist_ok=True)

h = history.history

# Training curves (on scaled targets)
plt.figure(figsize=(8,5))
plt.plot(h["loss"], label="Train Loss")
plt.plot(h["val_loss"], label="Val Loss")
plt.title("Loss over Epochs"); plt.xlabel("Epoch"); plt.ylabel("MSE/Huber (scaled)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"loss_{ts}.png"), dpi=150); plt.close()

plt.figure(figsize=(8,5))
plt.plot(h["mae"], label="Train MAE (scaled)")
plt.plot(h["val_mae"], label="Val MAE (scaled)")
plt.title("MAE over Epochs"); plt.xlabel("Epoch"); plt.ylabel("MAE (scaled)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"mae_scaled_{ts}.png"), dpi=150); plt.close()

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
X_all = pad_sequences([encode_seq(p) for p in test_df["prefix"]], maxlen=maxlen, padding="post", value=PAD_ID)
y_true_all = test_df["next_time_delta"].values.reshape(-1, 1)
y_pred_all = scaler.inverse_transform(model.predict(X_all, verbose=0))
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
artifact = wandb.Artifact("lstm_nt_model_bpi2012c", type="model")
artifact.add_file(CKPT_PATH)
run.log_artifact(artifact)

# %%
wandb.finish()
