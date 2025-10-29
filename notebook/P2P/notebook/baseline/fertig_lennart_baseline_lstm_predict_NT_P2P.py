# %% LSTM — Next-Time (NT) prediction
# - Temporal split by case start (uses processed splits from `data.loader`)
# - Inputs: tokenized activity prefixes (pre-padded to max_case_length) + 3 time features
#   (recent_time, latest_time, time_passed), each standardized with train-fitted scalers
# - Model: Token Embedding → (1–2) LSTM layers → Dropout
#          + small MLP on time features → Concatenate → Dense(1) (prediction in standardized space)
# - Target: next_time standardized (StandardScaler); train with LogCosh loss
# - Reporting: inverse-transform predictions/labels to days → MAE / MSE / RMSE (days)
# - Evaluation: per-k curves + macro averages; global scatter & error histogram; sample preds
# - Training: Adam(clipnorm), val_loss checkpointing, EarlyStopping, ReduceLROnPlateau
# - Logging/plots: W&B logging + headless matplotlib; fixed seeds for reproducibility

import os, sys, random, glob, ctypes
os.environ["MPLBACKEND"] = "Agg"  # headless matplotlib

# Preload libstdc++ on some HPC stacks (no-op if not needed)
prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
cands = glob.glob(os.path.join(prefix, "lib", "libstdc++.so.6*"))
if cands:
    try:
        mode = getattr(ctypes, "RTLD_GLOBAL", 0)  # ← use ctypes, not os
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
from wandb.integration.keras import WandbMetricsLogger

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from sklearn import metrics

# Data Pipeline
from data import loader
from data.constants import Task

# %% W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %% Config
DATASET = "P2P"

config = {
    # bookkeeping
    "dataset":                  DATASET,
    "checkpoint_path":          f"/tmp/best_lstm_nt_{DATASET}.weights.h5",
    "monitor_metric":           "val_loss",
    "monitor_mode":             "min",
    # optimization
    "learning_rate":            3e-4, 
    "clipnorm":                 1.0,
    "early_stop_patience":      7,
    "batch_size":               64,
    "epochs":                   90,
    # scheduler & early stop
    "early_stop_patience":      7,
    "reduce_lr_factor":         0.5,
    "reduce_lr_patience":       3,
    "min_lr":                   1e-6,
    # model scale
    "embed_dim":                256,
    "lstm_units_1":             256,        # 128 → 256 (20.09.)
    "lstm_units_2":             128,        # 64 → 128 (20.09.)
    "dropout":                  0.30,       # 0.20 → 0.30 (20.09.)
    "recurrent_dropout":        0.10,       # added (20.09.)
    "l2":                       1e-5,
}

# %%
config["seed"] = 41
tf.keras.utils.set_random_seed(config["seed"])
tf.config.experimental.enable_op_determinism()
random.seed(config["seed"])
np.random.seed(config["seed"])

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project=f"baseline_lstm_NT_{config['dataset']}",
    entity="privajet-university-of-mannheim",
    name=f"lstm_nt_{ts}",
    config=config
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

# %% Inputs
tok_in  = Input(shape=(max_case_length,), name="tokens")
time_in = Input(shape=(3,), name="time_feats")

x = Embedding(
    input_dim=vocab_size,
    output_dim=config["embed_dim"],
    mask_zero=True
)(tok_in)

x = LSTM(
    config["lstm_units_1"],
    return_sequences=(config["lstm_units_2"] > 0),
    dropout=config["dropout"],
    recurrent_dropout=config["recurrent_dropout"],
    kernel_regularizer=l2(config["l2"]),
    recurrent_initializer="orthogonal",
    recurrent_regularizer=l2(config["l2"])
)(x)
x = Dropout(config["dropout"])(x)

if config["lstm_units_2"] > 0:
    x = LSTM(
        config["lstm_units_2"],
        dropout=config["dropout"],
        recurrent_dropout=config["recurrent_dropout"],
        kernel_regularizer=l2(config["l2"]),
        recurrent_initializer="orthogonal",
        recurrent_regularizer=l2(config["l2"])
    )(x)
    x = Dropout(config["dropout"])(x)

# Time branch (small MLP, keeps it simple)
t = Dense(32, activation="relu", kernel_regularizer=l2(config["l2"])) (time_in)

# Fuse & head
h = Concatenate()([x, t])
h = Dense(64, activation="relu", kernel_regularizer=l2(config["l2"])) (h)
h = Dropout(config["dropout"]) (h)
out = Dense(1, activation="linear")(h)

# %% Model
model = Model(inputs=[tok_in, time_in], outputs=out)

model.compile(
    optimizer=Adam(learning_rate=config["learning_rate"], clipnorm=config["clipnorm"]),
    loss=tf.keras.losses.LogCosh(),
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
)

# %% Callbacks
checkpoint_cb = ModelCheckpoint(
    filepath=config["checkpoint_path"],
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
    [train_tok_x, train_time_x], train_y,
    validation_data=([val_tok_x, val_time_x], val_y),
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    callbacks=[checkpoint_cb, early_stop, reduce_lr, WandbMetricsLogger()],
    verbose=2
)

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
    y_scaled = model.predict([tok_x, time_x], verbose=0)
    y_days = float(y_scaler.inverse_transform(y_scaled)[0, 0])
    return max(0.0, y_days)

# %% Per-k evaluation (metrics in days)
k_vals, maes, mses, rmses, counts = [], [], [], [], []

for k in range(1, int(max_case_length) + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue

    sub_tok_x, sub_time_x, sub_y, _, _ = data_loader.prepare_data_next_time(
        subset, x_word_dict, max_case_length,
        time_scaler=time_scaler, y_scaler=y_scaler, shuffle=False
    )

    y_pred_scaled = model.predict([sub_tok_x, sub_time_x], verbose=0)
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

# %% Plots → disk
plot_dir = f"/ceph/lfertig/Thesis/notebook/{config['dataset']}/plots/Baselines/LSTM/NT"
os.makedirs(plot_dir, exist_ok=True)

h = history.history

# Training curves (log-space targets)
plt.figure(figsize=(8,5))
plt.plot(h["loss"], label="Train")
plt.plot(h["val_loss"], label="Validation")
plt.title("Loss over Epochs (log-space)")
plt.xlabel("Epoch"); plt.ylabel("Loss (Huber/MSE in log-space)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"loss_logspace_{ts}.png"), dpi=150); plt.close()

plt.figure(figsize=(8,5))
plt.plot(h["mae"], label="Train MAE (log-space)")
plt.plot(h["val_mae"], label="Val MAE (log-space)")
plt.title("MAE over Epochs (log-space)")
plt.xlabel("Epoch"); plt.ylabel("MAE (log-space)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"mae_logspace_{ts}.png"), dpi=150); plt.close()

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
y_pred_scaled_all = model.predict([test_tok_x, test_time_x], verbose=0)
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
    pred_scaled = model.predict([sub_tok_x, sub_time_x], verbose=0)
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

# %% Save checkpoint as W&B artifact
artifact = wandb.Artifact(f"lstm_nt_model_{config['dataset']}", type="model")
artifact.add_file(config["checkpoint_path"])
run.log_artifact(artifact)

# %%
wandb.finish()