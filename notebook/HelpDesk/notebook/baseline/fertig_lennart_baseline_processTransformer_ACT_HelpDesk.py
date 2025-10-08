# %%
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

from sklearn import metrics
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ProcessTransformer
from models import transformer

# %%
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %%
config = {
    # bookkeeping
    "checkpoint_path": "/tmp/best_transformer_act_HelpDesk.weights.h5",
    "monitor_metric":  "val_loss",
    "monitor_mode":    "min",
    # seq handling
    "pad_direction": "pre",
    "truncating":    "pre",
    "max_ctx":       30,        # 30 (27.09.)
    # optimization
    "learning_rate":  1e-5,     # 3e-4 (27.09.); 1e-4 (28.09.); 1e-3, 5e-4, 3e-4, 1e-4
    "weight_decay":   1e-5,     # 1e-4 (20.09.)
    "clipnorm":       1.0,
    "batch_size":     128,       # 32→64 helps stability here (20.09.)
    "epochs":         60,
    # scheduler & early stop
    "early_stop_patience": 18,   # 7→12→18 (01.10.)
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 3,
    "min_lr": 1e-6,
    # model scale
    "embed_dim": 256,           # 64→128 (try 256 if VRAM allows) (20.09.)
    "num_heads": 8,             # keep embed_dim % num_heads == 0 (20.09.)
    "ff_dim":    1024,          # 256→512 (try 1024 for embed_dim=256) (20.09.)
    "num_layers": 5,            # 2→3 (27.09.)
    # regularization
    "dropout": 0.20,            # attention/FF dropout (your block uses one 'rate') (20.09.)
    # metrics
    "topk": [3, 5]
}

# %%
config["seed"] = 41
tf.keras.utils.set_random_seed(config["seed"])
tf.config.experimental.enable_op_determinism()

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_processTransformer_ACT_HelpDesk",
    entity="privajet-university-of-mannheim",
    name=f"transformer_act_{ts}",
    config=config,
    resume="never",
    force=True
)

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
    ).astype("int32")
    y = frame["next_activity"].map(encode_label).to_numpy()
    return X, y

X_train, y_train = prepare_data(train_df)
X_val,   y_val   = prepare_data(val_df)
X_test,  y_test  = prepare_data(test_df)

# %%
class PatchedTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate):
        super().__init__()
        self.block = transformer.TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    def call(self, inputs, training=False):
        return self.block(inputs, training=training)

class PadMask(layers.Layer):
    """Return mask of shape (B, L, 1) where 1=token, 0=PAD(=0)."""
    def call(self, inputs):
        # inputs: (B, L) int32
        m = tf.cast(tf.not_equal(inputs, 0), tf.float32)
        return tf.expand_dims(m, axis=-1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)

class MaskedAverage(layers.Layer):
    """Return masked average over time: sum(x*mask)/sum(mask)."""
    def call(self, inputs):
        x, mask = inputs  # x: (B,L,D), mask: (B,L,1)
        x = x * mask
        sum_x = tf.reduce_sum(x, axis=1)                         # (B, D)
        denom = tf.clip_by_value(tf.reduce_sum(mask, axis=1), 1e-6, 1e9)  # (B,1)
        return sum_x / denom                                     # (B, D)
    def compute_output_shape(self, input_shapes):
        x_shape, _ = input_shapes
        return (x_shape[0], x_shape[-1])

def get_next_activity_model(max_case_length, vocab_size, output_dim, embed_dim, num_heads, ff_dim, num_layers, dropout):
    # IMPORTANT: dtype=int32 for token ids
    inputs = layers.Input(shape=(max_case_length,), dtype="int32")
    # Token + position embeddings from processtransformer
    x = transformer.TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
    # PAD mask
    mask = PadMask(name="pad_mask")(inputs)                       # (B,L,1)
    # Zero PADs before attention
    x = layers.Multiply(name="mask_before_attn")([x, mask])
    for li in range(num_layers):
        x = PatchedTransformerBlock(embed_dim, num_heads, ff_dim, rate=dropout)(x)
        # Zero PADs again (safety) and masked average pool
        x = layers.Multiply(name=f"mask_after_attn_{li}")([x, mask])
    x = MaskedAverage(name="masked_avg")([x, mask])               # (B,D)
    # Head
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(output_dim, activation="linear")(x)     # from_logits=True
    return tf.keras.Model(inputs=inputs, outputs=logits, name="next_activity_transformer")

model = get_next_activity_model(
    max_case_length=maxlen,
    vocab_size=vocab_size,               # includes PAD
    output_dim=num_classes,             # excludes PAD
    embed_dim=config["embed_dim"],
    num_heads=config["num_heads"],
    ff_dim=config["ff_dim"],
    num_layers=config["num_layers"],
    dropout=config["dropout"]
)

metrics_list = [
    tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")
]

optimizer = AdamW(learning_rate=config["learning_rate"],
                  weight_decay=config["weight_decay"],
                  clipnorm=config["clipnorm"])

loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics_list
)

# %%
checkpoint_cb = ModelCheckpoint(config["checkpoint_path"], save_weights_only=True,
                       monitor=config["monitor_metric"], mode=config["monitor_mode"], save_best_only=True, verbose=1)

early_stop = EarlyStopping(monitor=config["monitor_metric"], mode=config["monitor_mode"],
                     patience=config["early_stop_patience"], restore_best_weights=True, verbose=1)

reduce_lr  = ReduceLROnPlateau(monitor=config["monitor_metric"], mode=config["monitor_mode"],
                         factor=config["reduce_lr_factor"], patience=config["reduce_lr_patience"],
                         min_lr=config["min_lr"], verbose=1)

wandb_cb = WandbMetricsLogger()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    # class_weight={i: float(class_weights[i]) for i in range(num_classes)},
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    callbacks=[checkpoint_cb, early_stop, reduce_lr, WandbMetricsLogger()],
    verbose=2
)
model.load_weights(config["checkpoint_path"])

# %% Inference helper (top-k)
def predict_next(prefix, topk=5):
    x = pad_sequences([encode_prefix(prefix)], maxlen=maxlen,
                      padding=config["pad_direction"],
                      truncating=config["truncating"],
                      value=PAD_ID)
    logits = model.predict(x, verbose=0)[0]
    probs = tf.nn.softmax(logits).numpy()
    top_idx = probs.argsort()[-topk:][::-1]
    top_lbl = [inv_y[i] for i in top_idx]     # map ids -> labels via metadata
    return top_lbl[0], list(top_lbl), float(probs[top_idx[0]]), [float(probs[i]) for i in top_idx]

# %% Per-k loop over actual k values; compute macro averages; keep curves clean (no synthetic 'avg' point)
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

# %% Plots → disk
plot_dir = "/ceph/lfertig/Thesis/notebook/HelpDesk/plots/Baselines/Transformer/ACT"
os.makedirs(plot_dir, exist_ok=True)

# Loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"loss_{ts}.png"), dpi=150); plt.close()

# Accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"accuracy_{ts}.png"), dpi=150); plt.close()

# Acc/F1 vs k
if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, accuracies, marker='o', label='Accuracy')
    plt.title('Accuracy vs. Prefix Length (k)'); plt.xlabel('Prefix Length (k)'); plt.ylabel('Accuracy')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"acc_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, fscores, marker='o', label='F1 (weighted)')
    plt.title('F1 vs. Prefix Length (k)'); plt.xlabel('Prefix Length (k)'); plt.ylabel('F1 (weighted)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"f1_vs_k_{ts}.png"), dpi=150); plt.close()

print(f"Saved plots to: {plot_dir}")

# %% Log per-k curves + macro averages
wandb.log({
    "curves/k": k_vals,
    "curves/accuracy": accuracies,
    "curves/f1": fscores,
    "curves/precision": precisions,
    "curves/recall": recalls,
    "curves/counts": counts,
    "metrics/avg_accuracy": avg_accuracy,
    "metrics/avg_f1": avg_f1,
    "metrics/avg_precision": avg_precision,
    "metrics/avg_recall": avg_recall,
})

# %% Robust confusion matrix (avoid KeyError by normalizing strings + union classes)
def _norm(s): return str(s).strip()

y_pred_all = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true_lbl = [_norm(inv_y[i]) for i in y_test]
y_pred_lbl = [_norm(inv_y[i]) for i in y_pred_all]
cm_labels  = sorted(set(y_true_lbl) | set(y_pred_lbl))

try:
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true_lbl,
            preds=y_pred_lbl,
            class_names=cm_labels
        )
    })
except Exception as e:
    print("W&B confusion_matrix failed, falling back to static image:", e)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=cm_labels)
    plt.figure(figsize=(max(6, len(cm_labels)*0.6), max(5, len(cm_labels)*0.5)))
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title('Confusion Matrix (Transformer ACT)')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(ticks=range(len(cm_labels)), labels=cm_labels, rotation=90)
    plt.yticks(ticks=range(len(cm_labels)), labels=cm_labels)
    plt.tight_layout()
    cm_path = os.path.join(plot_dir, f"confusion_matrix_{ts}.png")
    plt.savefig(cm_path, dpi=150); plt.close()
    wandb.log({"cm_image": wandb.Image(cm_path)})

# %% Sample predictions (print + W&B table), like your LSTM ACT style
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

# %% Save weights as artifact
artifact = wandb.Artifact("transformer_act_model", type="model")
artifact.add_file(config["checkpoint_path"])
run.log_artifact(artifact)
# %%
wandb.finish()