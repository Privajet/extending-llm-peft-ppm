# %%
import os
os.environ["MPLBACKEND"] = "Agg"   # headless matplotlib
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
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy, SparseCategoricalAccuracy

from sklearn import metrics

# %%
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %%
config = {
    "checkpoint_path": "/tmp/best_lstm_act_HelpDesk.weights.h5",
    "monitor_metric": "val_loss",
    "monitor_mode": "min",
    "pad_direction": "pre",
    "truncating": "pre",
    "early_stop_patience":      7,
    "reduce_lr_factor":         0.5,
    "reduce_lr_patience":       3,
    "min_lr":                   1e-6,
    "max_ctx":                  30,
    "weight_decay":             1e-4,
    "learning_rate":            3e-4,       # 5e-4 → 3e-4 20.09.
    "batch_size":               64,
    "epochs":                   90,
    "embedding_dim":            256,        # 64 → 256 20.09.
    "lstm_units_1":             256,        # 128 → 256 20.09.
    "lstm_units_2":             128,        # 64 → 128 20.09.
    "dropout":                  0.30,       # 0.0 → 0.3 20.09
    "recurrent_dropout":        0.10,
    "l2":                       1e-5,       # 1e-5 → 1e-4 20.09.
    "clipnorm":                 1.0,
    "label_smoothing":          0.1       # 0.0 → 0.05–0.10 20.09
}

# %%
# init with project/entity/name, pass your config dict explicitly
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_lstm_ACT_HelpDesk",
    entity="privajet-university-of-mannheim",
    name=f"lstm_act_{ts}",
    config=config
)

CHECKPOINT_PATH = config["checkpoint_path"]
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

# %% Data
train_df = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_train.csv")
val_df   = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_val.csv")
test_df  = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_test.csv")

for d in (train_df, val_df, test_df):
    d.rename(columns={"next_act": "next_activity"}, inplace=True)
    d["prefix"] = d["prefix"].astype(str).str.split() # convert space-separated strings to lists

print(f"Train prefixes: {len(train_df)} - Validation prefixes: {len(val_df)} - Test prefixes: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# %%
PROC_DIR   = "/ceph/lfertig/Thesis/data/HelpDesk/processed"
META_PATH  = os.path.join(PROC_DIR, "metadata.json")

with open(META_PATH, "r") as f:
    meta = json.load(f)

x_word_dict = meta["x_word_dict"]  # includes [PAD]=0, [UNK]=1
y_word_dict = meta["y_word_dict"]  # label -> id (0..C-1)

vocab_size  = len(x_word_dict)
num_classes = len(y_word_dict)

PAD_ID = x_word_dict["[PAD]"]      # should be 0
UNK_ID = x_word_dict["[UNK]"]      # should be 1

# Inverse label mapping for pretty-printing predictions
inv_y = {v: k for k, v in y_word_dict.items()}

# Use real prefix lengths, capped by max_ctx
maxlen = min(config["max_ctx"], max(len(p) for p in train_df["prefix"]))

def encode_prefix(tokens):
    # tokens is a list like ["assign-seriousness", "take-in-charge-ticket", ...]
    return [x_word_dict.get(t, UNK_ID) for t in tokens]

def encode_label(act):
    return y_word_dict[act]

def prepare_data(frame):
    X = [encode_prefix(p) for p in frame["prefix"]]
    X = pad_sequences(
        X, maxlen=maxlen,
        padding=config["pad_direction"], truncating=config["truncating"], value=PAD_ID
    )
    y = frame["next_activity"].map(encode_label).to_numpy()
    return X, y

X_train, y_train = prepare_data(train_df)
X_val,   y_val   = prepare_data(val_df)
X_test,  y_test  = prepare_data(test_df)

# %%
layers = [
    Embedding(
        vocab_size, 
        config["embedding_dim"], 
        input_length=maxlen, 
        mask_zero=True
    ),
    LSTM(
        config["lstm_units_1"],
        return_sequences=(config["lstm_units_2"] > 0),
        dropout=config["dropout"],
        recurrent_dropout=config["recurrent_dropout"],
        kernel_regularizer=l2(config["l2"]),
        recurrent_regularizer=l2(config["l2"]),
        recurrent_initializer="orthogonal"
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
            recurrent_regularizer=l2(config["l2"]),
            recurrent_initializer="orthogonal"
        ),
        Dropout(config["dropout"])
    ]

layers += [Dense(num_classes, activation="linear")]

model = Sequential(layers)

optimizer = AdamW(
    learning_rate=config["learning_rate"],
    weight_decay=config["weight_decay"],
    clipnorm=config["clipnorm"]
)

metrics_list = [
    SparseCategoricalAccuracy(name="acc"),
    SparseTopKCategoricalAccuracy(k=3, name="top3"),
    SparseTopKCategoricalAccuracy(k=5, name="top5"),
]

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, label_smoothing=config["label_smoothing"])

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics_list
)

# %%
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

# %%
def predict_next(prefix_tokens, topk=5):
    x = pad_sequences([encode_prefix(prefix_tokens)], maxlen=maxlen,
                      padding=config["pad_direction"], truncating=config["truncating"], value=PAD_ID)
    logits = model.predict(x, verbose=0)[0]
    probs = tf.nn.softmax(logits).numpy()
    top_idx = probs.argsort()[-topk:][::-1]
    top_lbl = [inv_y[i] for i in top_idx]
    return top_lbl[0], list(top_lbl), float(probs[top_idx[0]]), [float(probs[i]) for i in top_idx]

# %%
k_vals, accuracies, fscores, precisions, recalls, counts = [], [], [], [], [], []

for k in sorted(test_df["k"].astype(int).unique()):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue
    X_t, y_t = prepare_data(subset)
    y_pred = np.argmax(model.predict(X_t, verbose=0), axis=1)
    acc = metrics.accuracy_score(y_t, y_pred)
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(y_t, y_pred, average="weighted", zero_division=0)
    k_vals.append(k); counts.append(len(y_t))
    accuracies.append(acc); fscores.append(f1); precisions.append(prec); recalls.append(rec)

avg_accuracy = float(np.mean(accuracies)) if accuracies else float("nan")
avg_f1       = float(np.mean(fscores))    if fscores    else float("nan")
avg_precision= float(np.mean(precisions)) if precisions else float("nan")
avg_recall   = float(np.mean(recalls))    if recalls    else float("nan")

print(f"Average accuracy across all prefixes:  {avg_accuracy:.4f}")
print(f"Average f-score across all prefixes:   {avg_f1:.4f}")
print(f"Average precision across all prefixes: {avg_precision:.4f}")
print(f"Average recall across all prefixes:    {avg_recall:.4f}") 

# %%
plot_dir = "/ceph/lfertig/Thesis/notebook/HelpDesk/plots/Baselines/LSTM/ACT"
os.makedirs(plot_dir, exist_ok=True)

h = history.history

# Loss
plt.figure(figsize=(8,5))
plt.plot(h["loss"], label="Train Loss")
plt.plot(h["val_loss"], label="Validation Loss")
plt.title("Loss over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"loss_{ts}.png"), dpi=150); plt.close()

# Accuracy
acc_key = "acc" if "acc" in h else ("sparse_categorical_accuracy" if "sparse_categorical_accuracy" in h else list(h.keys())[1])
plt.figure(figsize=(8,5))
plt.plot(h[acc_key], label="Train Accuracy")
plt.plot(h["val_" + acc_key], label="Validation Accuracy")
plt.title("Accuracy over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"accuracy_{ts}.png"), dpi=150); plt.close()

# Acc/F1 vs k
if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, accuracies, marker="o", label="Accuracy")
    plt.title("Accuracy vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("Accuracy")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"acc_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, fscores, marker="o", label="F1 (weighted)")
    plt.title("F1 vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("F1 (weighted)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"f1_vs_k_{ts}.png"), dpi=150); plt.close()

print(f"Saved plots to: {plot_dir}")

# %%
wandb.log({
    "curves/k": k_vals,
    "curves/counts": counts,
    "curves/accuracy": accuracies,
    "curves/f1": fscores,
    "curves/precision": precisions,
    "curves/recall": recalls,
    "metrics/avg_accuracy": avg_accuracy,
    "metrics/avg_f1": avg_f1,
    "metrics/avg_precision": avg_precision,
    "metrics/avg_recall": avg_recall,
})

# %%
y_pred_all = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true_str = [inv_y[int(i)] for i in y_test]
y_pred_str = [inv_y[int(i)] for i in y_pred_all]
cm_labels  = sorted(set(y_true_str) | set(y_pred_str))

try:
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true_str,
            preds=y_pred_str,
            class_names=cm_labels
        )
    })
except Exception as e:
    print("W&B confusion_matrix failed, will skip. Err:", e)

# %%
sample = test_df.sample(n=min(5, len(test_df)), random_state=42) if len(test_df) else test_df
table = wandb.Table(columns=["k", "prefix", "gold", "pred", "p_pred", "top5", "top5_p"])

for _, r in sample.iterrows():
    pred, top5, p_pred, top5_p = predict_next(r["prefix"], topk=5)
    print("Prefix:", " → ".join(r["prefix"]))
    print("Gold:  ", r["next_activity"])
    print(f"Pred:  {pred} ({p_pred:.3f})")
    print("Top-5:", top5)
    print("-"*60)
    table.add_data(
        r["k"],
        " → ".join(r["prefix"]),
        r["next_activity"],
        pred,
        p_pred,
        ", ".join(top5),
        ", ".join([f"{x:.3f}" for x in top5_p])
    )
wandb.log({"samples": table})

# %%
artifact = wandb.Artifact("lstm_act_model_helpdesk", type="model")
artifact.add_file(CHECKPOINT_PATH)
run.log_artifact(artifact)

# %%
wandb.finish()