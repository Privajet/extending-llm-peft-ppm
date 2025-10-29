# %%
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

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy

# ProcessTransformer
from data.models import transformer

# %%
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %%
# config = {
#     #"learning_rate": 1e-3,
#     "learning_rate": 5e-4, # [2e-4, 3e-4]
#     "batch_size": 32,
#     "epochs": 50, # 60
#     "embed_dim": 36, # [(48, 192)]
#     "num_heads": 4,
#     "ff_dim": 64 # [(64, 256)]
# }

# %%
# # A mit original LR
# config = {
#   "learning_rate": 5e-4,
#   "batch_size": 32,
#   "epochs": 60,
#   "embed_dim": 64,
#   "num_heads": 4,     # 64/4 = 16 per head
#   "ff_dim": 256       # ≈4× d_model
# }
# # Why: Your val-loss keeps falling; that suggests under-capacity more than overfit. Raising embed_dim and ff_dim (to the common 4× rule) should help. Slight LR drop + ~10 extra epochs lets the larger model settle.

# %%
# #A
# config = {
#   "learning_rate": 3e-4,
#   "batch_size": 32,
#   "epochs": 60,
#   "embed_dim": 64,
#   "num_heads": 4,     # 64/4 = 16 per head
#   "ff_dim": 256       # ≈4× d_model
# }
# #Why: Your val-loss keeps falling; that suggests under-capacity more than overfit. Raising embed_dim and ff_dim (to the common 4× rule) should help. Slight LR drop + ~10 extra epochs lets the larger model settle.

# %%
# #B: 
# config = {
#   "learning_rate": 2e-4,
#   "batch_size": 32,
#   "epochs": 80,
#   "embed_dim": 36,
#   "num_heads": 4,
#   "ff_dim": 144       # ~4×36 for a better FFN ratio
# }
# #Why: Your curves don’t look noisy; they look like they’d keep improving with more steps and a gentler LR. This mimics LR decay without changing code.

# %%
# #C
# config = {
#   "learning_rate": 3e-4,
#   "batch_size": 32,
#   "epochs": 60,
#   "embed_dim": 64,
#   "num_heads": 8,     # 64/8 = 8 per head
#   "ff_dim": 256
# }
# #Why: If long-range interactions matter, extra heads can help. Expect higher compute/memory than A.

# %% - Modified: A)
config = {
  "learning_rate": 3e-4,
  "batch_size": 32,
  "epochs": 80,
  "embed_dim": 64,
  "num_heads": 8,     # 64/8 = 8 per head
  "ff_dim": 256,
  "clipnorm": 1.0 
}
#Why: If long-range interactions matter, extra heads can help. Expect higher compute/memory than A.

# %%
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_processTransformer_ACT_BPIC20_DomesticDeclarations",
    entity="privajet-university-of-mannheim",
    name=f"transformer_act_{ts}",
    config=config,
    resume="never",
    force=True
)

# %%
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_bpic20_domestic.csv.gz")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)

# %% Build all prefixes (1..n-1)
prefix_data = []
for case_id, g in df.groupby("case:concept:name"):
    acts = g["concept:name"].tolist()
    for i in range(1, len(acts)):
        prefix_data.append({
            "case_id": case_id,
            "prefix": acts[:i],
            "next_activity": acts[i],
            "k": i
        })

processTransformer_act_prefix_df = pd.DataFrame(prefix_data)

# %% Temporal split by case start
case_start = df.groupby("case:concept:name")["time:timestamp"].min().reset_index()
case_start = case_start.sort_values("time:timestamp")
case_ids = case_start["case:concept:name"].tolist()

n_total = len(case_ids)
n_train = int(n_total * 0.8)
n_val   = int(n_train * 0.2)

train_ids = case_ids[:n_train - n_val]
val_ids   = case_ids[n_train - n_val:n_train]
test_ids  = case_ids[n_train:]

train_df = processTransformer_act_prefix_df[processTransformer_act_prefix_df["case_id"].isin(train_ids)].reset_index(drop=True)
val_df   = processTransformer_act_prefix_df[processTransformer_act_prefix_df["case_id"].isin(val_ids)].reset_index(drop=True)
test_df  = processTransformer_act_prefix_df[processTransformer_act_prefix_df["case_id"].isin(test_ids)].reset_index(drop=True)

print(f"Train prefixes: {len(train_df)} - Validation prefixes: {len(val_df)} - Test prefixes: {len(test_df)}")

# %%
encoder = LabelEncoder().fit(df["concept:name"])
PAD_ID = 0

def enc(seq): 
    return encoder.transform(seq) + 1  # +1 offset -> PAD=0 reserved

vocab_size = len(encoder.classes_) + 1   # +1 for PAD
maxlen = processTransformer_act_prefix_df["k"].max()

def prepare_data(df_part, maxlen):
    X = [enc(p) for p in df_part["prefix"]]
    X = pad_sequences(X, maxlen=maxlen, padding="post", value=PAD_ID)  # post-pad (match LSTM)
    y = encoder.transform(df_part["next_activity"])                    # class ids (no PAD)
    return X, y


X_train, y_train = prepare_data(train_df, maxlen)
X_val,   y_val   = prepare_data(val_df,   maxlen)
X_test,  y_test  = prepare_data(test_df,  maxlen)

# %%
def prepare_data(df_part, maxlen):
    X = [enc(p) for p in df_part["prefix"]]
    X = pad_sequences(X, maxlen=maxlen, padding="post", value=PAD_ID).astype("int32")
    y = encoder.transform(df_part["next_activity"])  # class ids (no PAD)
    return X, y

class PatchedTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
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

def get_next_activity_model(max_case_length, vocab_size, output_dim, embed_dim, num_heads, ff_dim):
    # IMPORTANT: dtype=int32 for token ids
    inputs = layers.Input(shape=(max_case_length,), dtype="int32")

    # Token + position embeddings from processtransformer
    x = transformer.TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)

    # PAD mask
    mask = PadMask(name="pad_mask")(inputs)                       # (B,L,1)

    # Zero PADs before attention
    x = layers.Multiply(name="mask_before_attn")([x, mask])
    x = PatchedTransformerBlock(embed_dim, num_heads, ff_dim)(x)
    # Zero PADs again (safety) and masked average pool
    x = layers.Multiply(name="mask_after_attn")([x, mask])
    x = MaskedAverage(name="masked_avg")([x, mask])               # (B,D)

    # Head
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    logits = layers.Dense(output_dim, activation="linear")(x)     # from_logits=True
    return tf.keras.Model(inputs=inputs, outputs=logits, name="next_activity_transformer")

model = get_next_activity_model(
    max_case_length=maxlen,
    vocab_size=vocab_size,               # includes PAD
    output_dim=len(encoder.classes_),    # excludes PAD
    embed_dim=config["embed_dim"],
    num_heads=config["num_heads"],
    ff_dim=config["ff_dim"]
)

metrics_list = [
    tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy"),
    SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
    SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
]
model.compile(
    optimizer=Adam(learning_rate=config["learning_rate"], clipnorm=config["clipnorm"]),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=metrics_list
)

# %%
ckpt_path = "/tmp/best_transformer_act_BPIC20_DomesticDeclarations.weights.h5"
checkpoint_cb = ModelCheckpoint(
    filepath=ckpt_path,
    save_weights_only=True,
    monitor="val_sparse_categorical_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

wandb_cb = WandbMetricsLogger()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    callbacks=[checkpoint_cb, wandb_cb],
    verbose=2
)

model.load_weights(ckpt_path)

# %% Inference helper (top-k, like LSTM)
def predict_next(prefix, topk=5):
    x = pad_sequences([enc(prefix)], maxlen=maxlen, padding="post", value=PAD_ID)
    logits = model.predict(x, verbose=0)[0]
    probs = tf.nn.softmax(logits).numpy()
    top_idx = probs.argsort()[-topk:][::-1]
    top_lbl = encoder.inverse_transform(top_idx)
    return top_lbl[0], list(top_lbl), float(probs[top_idx[0]]), [float(probs[i]) for i in top_idx]

# %% Per-k loop over actual k values; compute macro averages; keep curves clean (no synthetic 'avg' point)
k_vals, accuracies, fscores, precisions, recalls, counts = [], [], [], [], [], []

for i in range(1, maxlen + 1):
    subset = test_df[test_df["k"] == i]
    if subset.empty:
        continue

    X_t, y_t = prepare_data(subset, maxlen)
    y_pred = np.argmax(model.predict(X_t, verbose=0), axis=1)

    acc = metrics.accuracy_score(y_t, y_pred)
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(
        y_t, y_pred, average="weighted", zero_division=0
    )

    k_vals.append(i); counts.append(len(y_t))
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
plot_dir = "/ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/plots/Baselines/Transformer/ACT"
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
plt.plot(history.history['sparse_categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
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
y_true_lbl = [_norm(x) for x in encoder.inverse_transform(y_test)]
y_pred_lbl = [_norm(x) for x in encoder.inverse_transform(y_pred_all)]
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
artifact.add_file(ckpt_path)
run.log_artifact(artifact)

# %%
wandb.finish()