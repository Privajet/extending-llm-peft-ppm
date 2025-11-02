# %% LSTM — Next-Activity (ACT) prediction
# - Temporal split by case start (uses processed splits from `data.loader`)
# - Prefix of length k (tokenized activities) predicts the next activity class
# - Token Embedding → (1–2) LSTM layers → Dropout → Linear logits over |A|
# - Trained with SparseCategoricalCrossentropy(from_logits=True)
# - Metrics: Accuracy, Top-3, Top-5 (monitored: val_sparse_categorical_accuracy)
# - W&B logging + headless matplotlib plots

import os, sys, random, glob, ctypes
os.environ["MPLBACKEND"] = "Agg"   # headless matplotlib

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy, SparseCategoricalAccuracy

from sklearn import metrics

# Data Pipeline
from data import loader
from data.constants import Task

# %%
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %%
DATASET = "HelpDesk"

config = {
    # bookkeeping
    "dataset":                  DATASET,
    "checkpoint_path":          f"/tmp/best_lstm_act_{DATASET}.weights.h5",
    "monitor_metric":           "val_sparse_categorical_accuracy",
    "monitor_mode":             "max",
    # optimization
    "learning_rate":            3e-4,       # 5e-4 → 3e-4 20.09.
    "batch_size":               12,
    "epochs":                   100,
    # scheduler & early stop
    "early_stop_patience":      7,
    "reduce_lr_factor":         0.5,
    "reduce_lr_patience":       3,
    "min_lr":                   1e-6,
    # model scale / regularization
    "embed_dim":                256,        # 64 → 256 20.09.
    "lstm_units_1":             256,        # 128 → 256 20.09.
    "lstm_units_2":             128,        # 64 → 128 20.09.
    "dropout":                  0.30,       # 0.0 → 0.3 20.09
    "recurrent_dropout":        0.00,
    "l2":                       1e-5,       # 1e-4 → 1e-5 20.09.
    "clipnorm":                 1.0
}

# %%
config["seed"] = 41
tf.keras.utils.set_random_seed(config["seed"])
tf.config.experimental.enable_op_determinism()
random.seed(config["seed"])
np.random.seed(config["seed"])

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project=f"baseline_lstm_ACT_{config['dataset']}",
    entity="privajet-university-of-mannheim",
    name=f"lstm_act_{ts}",
    config=config
)

# %%
data_loader = loader.LogsDataLoader(name=config['dataset'])

(train_df, test_df, val_df,
 x_word_dict, y_word_dict,
 max_case_length, vocab_size,
 num_output) = data_loader.load_data(task=Task.NEXT_ACTIVITY)

wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

X_train, y_train = data_loader.prepare_data_next_activity(train_df, x_word_dict, y_word_dict, max_case_length)
X_val,   y_val   = data_loader.prepare_data_next_activity(val_df,   x_word_dict, y_word_dict, max_case_length, shuffle=False)
X_test,  y_test  = data_loader.prepare_data_next_activity(test_df,  x_word_dict, y_word_dict, max_case_length)

inv_y = {v: k for k, v in y_word_dict.items()}

# %%
layers = [
    Embedding(
        vocab_size, 
        config["embed_dim"], 
        input_length=max_case_length, 
        mask_zero=True
    ),
    LSTM(
        config["lstm_units_1"],
        return_sequences=(config["lstm_units_2"] > 0),
        dropout=config["dropout"],
        recurrent_dropout=config["recurrent_dropout"],
        kernel_regularizer=l2(config["l2"]),
        recurrent_regularizer=l2(config["l2"]),
        recurrent_initializer="orthogonal",
        use_cudnn=False
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
            recurrent_initializer="orthogonal",
            use_cudnn=False
        ),
        Dropout(config["dropout"])
    ]

layers += [Dense(num_output, activation="linear")]

# %%
model = Sequential(layers)

metrics_list = [
    SparseCategoricalAccuracy(),
    SparseTopKCategoricalAccuracy(k=3, name="top3"),
    SparseTopKCategoricalAccuracy(k=5, name="top5"),
]

model.compile(
    optimizer=Adam(learning_rate=config["learning_rate"], clipnorm=config["clipnorm"]),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=metrics_list
)

# %%
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
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    shuffle=True,
    callbacks=[checkpoint_cb, WandbMetricsLogger()],
    verbose=2
)

# %% Inference helper (top-k)
def predict_next(prefix_str: str, topk=5):
    df1 = pd.DataFrame([{
        "prefix":   prefix_str,
        "next_act": next(iter(y_word_dict.keys()))  # dummy,
    }])
    tok_x, _ = data_loader.prepare_data_next_activity(
        df1, x_word_dict, y_word_dict, max_case_length, shuffle=False
    )
    logits = model.predict(tok_x, verbose=0)[0]
    probs  = tf.nn.softmax(logits).numpy()

    top_idx = probs.argsort()[-topk:][::-1]
    top_lbl = [inv_y[i] for i in top_idx]
    top_prb = [float(probs[i]) for i in top_idx]
    
    return top_lbl[0], top_lbl, top_prb[0], top_prb

# %% Per-k loop over actual k values; compute macro averages over k; micro Accuracy
k_vals, accuracies, fscores, precisions, recalls, counts = [], [], [], [], [], []

for i in sorted(test_df["k"].astype(int).unique()):
    subset = test_df[test_df["k"] == i]
    if len(subset) > 0:
        test_token_x, test_token_y = data_loader.prepare_data_next_activity(subset, x_word_dict, y_word_dict, max_case_length)
        y_pred = np.argmax(model.predict(test_token_x, verbose=0), axis=1)
        accuracy = metrics.accuracy_score(test_token_y, y_pred)
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(test_token_y, y_pred, average="weighted", zero_division=0)
        
        k_vals.append(i)
        counts.append(len(test_token_y))
        accuracies.append(accuracy)
        fscores.append(fscore)
        precisions.append(precision)
        recalls.append(recall)

avg_accuracy = float(np.mean(accuracies)) if accuracies else float("nan")
avg_f1 = float(np.mean(fscores)) if fscores else float("nan")
avg_precision = float(np.mean(precisions)) if precisions else float("nan")
avg_recall = float(np.mean(recalls)) if recalls else float("nan")

print(f"Average accuracy across all prefixes:  {avg_accuracy:.4f}")
print(f"Average f-score across all prefixes:   {avg_f1:.4f}")
print(f"Average precision across all prefixes: {avg_precision:.4f}")
print(f"Average recall across all prefixes:    {avg_recall:.4f}") 

# Micro (global) accuracy over all test prefixes
y_pred_all = np.argmax(model.predict(X_val), axis=1)
micro_acc_val = metrics.accuracy_score(y_val, y_pred_all)
print(f"[VAL]  Micro (global) accuracy: {micro_acc_val:.4f}")

# Micro (global) accuracy over all test prefixes
y_pred_all = np.argmax(model.predict(X_test), axis=1)
micro_acc = metrics.accuracy_score(y_test, y_pred_all)
print(f"[TEST] Micro (global) accuracy: {micro_acc:.4f}")

# %% Plots → disk
plot_dir = f"/ceph/lfertig/Thesis/notebook/{config['dataset']}/plots/Baselines/LSTM/ACT"
os.makedirs(plot_dir, exist_ok=True)

# Loss
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"loss_{ts}.png"), dpi=150); plt.close()

# Accuracy
acc_key = "acc" if "acc" in history.history else ("sparse_categorical_accuracy" if "sparse_categorical_accuracy" in history.history else list(history.history.keys())[1])
plt.figure(figsize=(8,5))
plt.plot(history.history[acc_key], label="Train Accuracy")
plt.plot(history.history["val_" + acc_key], label="Validation Accuracy")
plt.title("Accuracy over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
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

# %% Robust confusion matrix (avoid KeyError by normalizing strings + union classes)
def _norm(s): return str(s).strip()

y_pred_all = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true_lbl = [_norm(inv_y[i]) for i in y_test]
y_pred_lbl = [_norm(inv_y[i]) for i in y_pred_all]
cm_labels = sorted({str(x) for x in y_true_lbl} | {str(x) for x in y_pred_lbl})

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
    prefix_str = str(r["prefix"])
    prefix_tokens = prefix_str.split()
    pred, top5, p_pred, top5_p = predict_next(prefix_str, topk=5)
    
    gold = str(r["next_act"])
    prefix_pretty = " → ".join(prefix_tokens)
    
    print("Prefix:", prefix_pretty)
    print("Gold:  ", gold)
    print(f"Pred:  {pred} ({p_pred:.3f})")
    print("Top-5:", top5)
    print("-"*60)
    table.add_data(
        r["k"],
        prefix_pretty,
        gold,
        pred,
        p_pred,
        ", ".join(top5),
        ", ".join([f"{x:.3f}" for x in top5_p])
    )
wandb.log({"samples": table})

# %% Save weights as artifact
artifact = wandb.Artifact(f"lstm_act_model_{config['dataset']}", type="model")
artifact.add_file(config["checkpoint_path"])
run.log_artifact(artifact)

# %%
wandb.finish()