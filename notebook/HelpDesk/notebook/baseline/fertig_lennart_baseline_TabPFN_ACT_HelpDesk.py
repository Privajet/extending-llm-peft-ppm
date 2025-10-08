# %%
import os
# Debug/compat env (must be set before importing torch/matplotlib)
os.environ["PYTORCH_SDP_BACKEND"] = "math"      # force non-fused SDPA globally
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"        # surface CUDA errors immediately
os.environ["MPLBACKEND"] = "Agg"                # headless matplotlib
import json
import sys, random, logging, joblib
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import torch

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass

import wandb
from sklearn import metrics
from tabpfn import TabPFNClassifier
try:
    from tabpfn_extensions.many_class import ManyClassClassifier
except Exception:
    ManyClassClassifier = None
# %%
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %%
config = {
    # bookkeeping
    # seq handling
    "pad_direction": "pre",
    "truncating":    "pre",
    "max_ctx":       30,
    # TabPFN
    "sample_size":   10000,  # downsample train for speed/stability; set None to disable
}

# %%
config["seed"] = 42
SEED   = config["seed"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config["device"] = str(DEVICE)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)
log.info("PyTorch: %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available():
    log.info("GPU: %s", torch.cuda.get_device_name(0))
print("Using device:", DEVICE)

# %%
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_tabpfn_ACT_HelpDesk",
    entity="privajet-university-of-mannheim",
    name=f"tabpfn_ACT_{ts}",
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

MAX_CTX = config["max_ctx"]
maxlen = min(MAX_CTX, max(len(p) for p in train_df["prefix"]))

def encode_prefix(tokens):
    return [x_word_dict.get(t, UNK_ID) for t in tokens]

def pad_trunc_pre(ids, L, pad_id):
    s = ids[-L:]                        # truncating='pre'
    return [pad_id]*(L - len(s)) + s    # padding='pre'

def encode_label(act):
    return y_word_dict[act]

def prepare_tabpfn(frame):
    X = [pad_trunc_pre(encode_prefix(p), maxlen, PAD_ID) for p in frame["prefix"]]
    y = frame["next_activity"].map(encode_label).to_numpy(dtype=np.int64)
    X = np.asarray(X, dtype=np.float32)
    return X, y

X_train, y_train = prepare_tabpfn(train_df)
X_val,   y_val   = prepare_tabpfn(val_df)
X_test,  y_test  = prepare_tabpfn(test_df)

wandb.log({
    "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
    "maxlen": maxlen, "num_classes": len(y_word_dict), "vocab_size": len(x_word_dict)
})

# %%
use_manyclass = (ManyClassClassifier is not None) and (len(y_word_dict) > 10)
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
          alphabet_size=min(10, len(y_word_dict)),
          n_estimators_redundancy=3,
          random_state=config["seed"],
          verbose=1
       ) if use_manyclass else base_clf)

# Downsample to keep TabPFN stable/fast on large datasets
SAMPLE_SIZE = config["sample_size"]
if SAMPLE_SIZE and len(X_train) > SAMPLE_SIZE:
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(X_train), size=SAMPLE_SIZE, replace=False)
    X_train_fit = X_train[sel]
    y_train_fit = y_train[sel]
    log.info("Downsampled train from %d to %d for TabPFN", len(X_train), len(X_train_fit))
    wandb.log({"train_downsampled_to": int(len(X_train_fit))})
else:
    X_train_fit, y_train_fit = X_train, y_train

clf.fit(X_train_fit, y_train_fit)

def predict_labels(model, X):
    """Return (pred, proba_or_None)."""
    try:
        proba = model.predict_proba(X)
        return proba.argmax(axis=1), proba
    except Exception:
        return model.predict(X), None

# %%
pred_val, _ = predict_labels(clf, X_val)
acc_val = metrics.accuracy_score(y_val, pred_val)
prec_val, rec_val, f1_val, _ = metrics.precision_recall_fscore_support(
    y_val, pred_val, average="weighted", zero_division=0
)
wandb.log({
    "val/accuracy": float(acc_val),
    "val/precision": float(prec_val),
    "val/recall": float(rec_val),
    "val/f1": float(f1_val)
})

# %%
# compute k from actual prefix length (capped at maxlen)
test_k = np.array([min(len(p), maxlen) for p in test_df["prefix"]], dtype=np.int32)

# use your helper to get labels (and probs if supported)
pred_test, _ = predict_labels(clf, X_test)

k_vals, accuracies, fscores, precisions, recalls, counts = [], [], [], [], [], []
for k in range(1, maxlen + 1):
    idx = np.where(test_k == k)[0]
    if len(idx) == 0: 
        continue
    yt, yp = y_test[idx], pred_test[idx]
    acc = metrics.accuracy_score(yt, yp)
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
    k_vals.append(k); counts.append(len(idx))
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
plot_dir = "/ceph/lfertig/Thesis/notebook/HelpDesk/plots/Baselines/TabPFN/ACT"
os.makedirs(plot_dir, exist_ok=True)

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
y_true_lbl = [inv_y[i] for i in y_test]
y_pred_lbl = [inv_y[i] for i in pred_test]
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
    log.warning("W&B CM failed: %s. Falling back to static CM image.", e)
    cm = metrics.confusion_matrix(y_true_lbl, y_pred_lbl, labels=cm_labels)
    plt.figure(figsize=(max(6, len(cm_labels)*0.6), max(5, len(cm_labels)*0.5)))
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title('Confusion Matrix (TabPFN ACT)'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(range(len(cm_labels)), cm_labels, rotation=90)
    plt.yticks(range(len(cm_labels)), cm_labels)
    plt.tight_layout()
    cm_path = f"/tmp/cm_tabpfn_ACT_{ts}.png"
    plt.savefig(cm_path, dpi=150); plt.close()
    wandb.log({"cm_image": wandb.Image(cm_path)})

# %%
def topk_accuracy(y_true, proba, k=3):
    topk = np.argpartition(proba, -k, axis=1)[:, -k:]
    hits = sum(y_true[i] in topk[i] for i in range(len(y_true)))
    return hits / len(y_true) if len(y_true) else float("nan")

_, proba_test = predict_labels(clf, X_test)
if proba_test is not None:
    top3 = topk_accuracy(y_test, proba_test, k=3)
    top5 = topk_accuracy(y_test, proba_test, k=5)
    print(f"Top-3 acc: {top3:.4f} | Top-5 acc: {top5:.4f}")
    wandb.log({"metrics/top3_acc": float(top3), "metrics/top5_acc": float(top5)})

# %% Replace the sample-table block with this:
def encode_one(prefix_tokens):
    return np.asarray([pad_trunc_pre(encode_prefix(prefix_tokens), maxlen, PAD_ID)], dtype=np.float32)

sample = test_df.sample(n=min(5, len(test_df)), random_state=42) if len(test_df) else test_df
table = wandb.Table(columns=["k", "prefix", "gold", "pred", "p_pred", "top5", "top5_p"])
for _, r in sample.iterrows():
    x = encode_one(r["prefix"])
    try:
        proba = clf.predict_proba(x)[0]
        top_idx = np.argsort(proba)[-5:][::-1]
        pred_id = int(top_idx[0])
        p_pred  = float(proba[pred_id])
        top5_lbl = [inv_y[int(i)] for i in top_idx]
        top5_p   = [float(proba[i]) for i in top_idx]
    except Exception:
        y_hat = clf.predict(x)
        pred_id = int(y_hat[0]); p_pred = float("nan")
        top5_lbl = [inv_y[pred_id]]
        top5_p   = [float("nan")]

    pred_lbl = inv_y[pred_id]
    print("Prefix:", " → ".join(r["prefix"]))
    print("Gold:  ", r["next_activity"])
    print(f"Pred:  {pred_lbl} ({p_pred:.3f})")
    print("Top-5:", list(top5_lbl))
    print("-"*60)

    table.add_data(
        min(len(r["prefix"]), maxlen),   # was r["k"]
        " → ".join(r["prefix"]),
        r["next_activity"],
        pred_lbl,
        p_pred,
        ", ".join([str(x) for x in top5_lbl]),
        ", ".join([f"{p:.3f}" for p in top5_p])
)
wandb.log({"samples": table})

# %%
ts_dir = f"/tmp/tabpfn_ACT_{ts}"
os.makedirs(ts_dir, exist_ok=True)
joblib.dump(clf, os.path.join(ts_dir, "tabpfn_model.pkl"))
with open(os.path.join(ts_dir, "meta.json"), "w") as f:
    json.dump({"maxlen": int(maxlen), "vocab_size": int(len(x_word_dict))}, f)
joblib.dump({"inv_y": inv_y, "y_word_dict": y_word_dict}, os.path.join(ts_dir, "label_map.pkl"))

artifact = wandb.Artifact(name=f"tabpfn_ACT_artifacts_{ts}", type="model")
artifact.add_dir(ts_dir)
run.log_artifact(artifact)

# %%
wandb.finish()