# %%
import os
os.environ["MPLBACKEND"] = "Agg"   # headless matplotlib

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
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy, SparseCategoricalAccuracy

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# %%
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %%
config = {
    "learning_rate": 5e-4,
    "batch_size": 32,
    "epochs": 90,
    "embedding_dim": 64,
    "lstm_units_1": 128,
    "lstm_units_2": 64,     # set 0 to disable 2nd layer
    "dropout": 0.20,
    "l2": 1e-5,
    "clipnorm": 1.0,
    "label_smoothing": 0.0
}

# %%
# init with project/entity/name, pass your config dict explicitly
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_lstm_ACT_BPIC20_DomesticDeclarations",
    entity="privajet-university-of-mannheim",
    name=f"lstm_act_{ts}",
    config=config
)

CHECKPOINT_PATH = "/tmp/best_lstm_act_BPIC20_DomesticDeclarations.weights.h5"
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

# %%
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_bpic20_domestic.csv.gz")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)

# %%
prefix_rows = []
for case_id, g in df.groupby("case:concept:name"):
    acts = g["concept:name"].tolist()
    for i in range(1, len(acts)):
        prefix_rows.append({
            "case_id": case_id,
            "prefix":  acts[:i],
            "k":       i,
            "next_activity": acts[i]
        })
act_df = pd.DataFrame(prefix_rows)

# %%
case_start = df.groupby("case:concept:name")["time:timestamp"].min().reset_index()
case_start = case_start.sort_values("time:timestamp")
case_ids   = case_start["case:concept:name"].tolist()

n_total = len(case_ids)
n_train = int(n_total * 0.8)
n_val   = int(n_train * 0.2)

train_ids = case_ids[:n_train - n_val]
val_ids   = case_ids[n_train - n_val:n_train]
test_ids  = case_ids[n_train:]

train_df = act_df[act_df["case_id"].isin(train_ids)].reset_index(drop=True)
val_df   = act_df[act_df["case_id"].isin(val_ids)].reset_index(drop=True)
test_df  = act_df[act_df["case_id"].isin(test_ids)].reset_index(drop=True)

print(f"Train prefixes: {len(train_df)} - Val: {len(val_df)} - Test: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

print(f"Train cases: {len(train_df)} - Validation cases: {len(val_df)} - Test cases: {len(test_df)}")

# %%
le = LabelEncoder().fit(df["concept:name"])
PAD_ID = 0
def encode_seq(seq): return le.transform(seq) + 1

vocab_size = len(le.classes_) + 1  # +1 for PAD
maxlen = act_df["k"].max()

def prepare_data(frame):
    X = [encode_seq(p) for p in frame["prefix"]]
    X = pad_sequences(X, maxlen=maxlen, padding="post", value=PAD_ID)  # <-- was "pre"
    y = le.transform(frame["next_activity"])
    return X, y

X_train, y_train = prepare_data(train_df)
X_val,   y_val   = prepare_data(val_df)
X_test,  y_test  = prepare_data(test_df)

# %%
layers = [
    Embedding(
        input_dim=vocab_size,
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

layers += [Dense(len(le.classes_), activation="linear")]

model = Sequential(layers)

metrics_list = [
    SparseCategoricalAccuracy(name="acc"),
    SparseTopKCategoricalAccuracy(k=3, name="top3"),
    SparseTopKCategoricalAccuracy(k=5, name="top5")
]

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=Adam(learning_rate=config["learning_rate"], clipnorm=config["clipnorm"]),
    loss=loss,
    metrics=metrics_list
)

# %%
checkpoint_cb = ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    save_weights_only=True,
    monitor="val_acc",
    save_best_only=True,
    mode="max",
    verbose=1
)
early_stop = EarlyStopping(
    monitor="val_acc",
    patience=7,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_acc",
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
if not os.path.exists(CHECKPOINT_PATH):
    model.save_weights(CHECKPOINT_PATH)
model.load_weights(CHECKPOINT_PATH)

# %%
def predict_next(prefix_tokens, topk=5):
    x = pad_sequences([encode_seq(prefix_tokens)], maxlen=maxlen, padding="post", value=PAD_ID)
    logits = model.predict(x, verbose=0)[0]
    probs = tf.nn.softmax(logits).numpy()
    top_idx = probs.argsort()[-topk:][::-1]
    top_lbl = le.inverse_transform(top_idx)
    return top_lbl[0], list(top_lbl), float(probs[top_idx[0]]), [float(probs[i]) for i in top_idx]

# %%
k_vals, counts = [], []
accuracies, precisions, recalls, fscores = [], [], [], []

for k in range(1, maxlen + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue
    X_t, y_t = prepare_data(subset)
    y_pred = np.argmax(model.predict(X_t, verbose=0), axis=1)

    acc = metrics.accuracy_score(y_t, y_pred)
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(
        y_t, y_pred, average="weighted", zero_division=0
    )

    k_vals.append(k); counts.append(len(y_t))
    accuracies.append(acc); precisions.append(prec); recalls.append(rec); fscores.append(f1)

# Macro averages across k-bins (do NOT append to curves)
avg_acc = float(np.mean(accuracies)) if accuracies else float("nan")
avg_f1  = float(np.mean(fscores))    if fscores    else float("nan")
avg_p   = float(np.mean(precisions)) if precisions else float("nan")
avg_r   = float(np.mean(recalls))    if recalls    else float("nan")

print(f"Average accuracy across all prefixes:  {avg_acc:.4f}")
print(f"Average f-score across all prefixes:   {avg_f1:.4f}")
print(f"Average precision across all prefixes: {avg_p:.4f}")
print(f"Average recall across all prefixes:    {avg_r:.4f}")

# %%
plot_dir = "/ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/plots/Baselines/LSTM/ACT"
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
    "metrics/avg_accuracy": avg_acc,
    "metrics/avg_f1": avg_f1,
    "metrics/avg_precision": avg_p,
    "metrics/avg_recall": avg_r,
})

# %%
y_pred_all = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true_str = [str(s).strip() for s in le.inverse_transform(y_test)]
y_pred_str = [str(s).strip() for s in le.inverse_transform(y_pred_all)]
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
artifact = wandb.Artifact("lstm_act_model", type="model")
artifact.add_file(CHECKPOINT_PATH)
run.log_artifact(artifact)

# %%
wandb.finish()