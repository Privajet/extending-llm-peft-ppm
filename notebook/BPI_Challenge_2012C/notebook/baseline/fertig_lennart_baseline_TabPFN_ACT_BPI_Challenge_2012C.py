# %%
import os
# Debug/compat env (must be set before importing torch/matplotlib)
os.environ["PYTORCH_SDP_BACKEND"] = "math"      # force non-fused SDPA globally
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"        # surface CUDA errors immediately
os.environ["MPLBACKEND"] = "Agg"                # headless matplotlib

import sys, random, logging, joblib

# matplotlib picks up MPLBACKEND from env; no need to call matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
# from pathlib import Path  # <- remove if unused
import numpy as np
import pandas as pd

import torch
# Disable Flash / fused SDPA kernels that can crash on some GPUs/driver combos
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass

import wandb
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from tabpfn import TabPFNClassifier

# %%
SEED = 42
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# %%
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

CFG = dict(device=str(DEVICE), sample_size=None, random_state=SEED)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_tabpfn_ACT_BPI_Challenge_2012C",
    entity="privajet-university-of-mannheim",
    name=f"tabpfn_ACT_{ts}",
    config=CFG,
    resume="never",
    force=True
)

# %%
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_bpi_challenge.csv.gz")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)

# Build ACT prefixes (1..n-1)
records = []
for case_id, g in df.groupby("case:concept:name"):
    acts = g["concept:name"].tolist()
    for i in range(1, len(acts)):
        records.append({
            "case_id": case_id,
            "prefix": acts[:i],
            "next_activity": acts[i],
            "k": i,
            "case_start": g["time:timestamp"].iloc[0],
        })
act_df = pd.DataFrame(records)
run.summary["total_prefixes"] = len(act_df)

# Temporal split by case start
case_start = (df.groupby("case:concept:name")["time:timestamp"]
                .min().reset_index().sort_values("time:timestamp"))
case_ids_sorted = case_start["case:concept:name"].tolist()

valid_ids = set(act_df["case_id"])
case_ids = [cid for cid in case_ids_sorted if cid in valid_ids]

n_total = len(case_ids)
n_train = int(0.8 * n_total)
n_val   = int(0.2 * n_train)

train_ids = case_ids[: n_train - n_val]
val_ids   = case_ids[n_train - n_val : n_train]
test_ids  = case_ids[n_train : ]

train_df = act_df[act_df["case_id"].isin(train_ids)].reset_index(drop=True)
val_df   = act_df[act_df["case_id"].isin(val_ids)  ].reset_index(drop=True)
test_df  = act_df[act_df["case_id"].isin(test_ids) ].reset_index(drop=True)

wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# %% Build a global category list: PAD first (will map to 0), then all activities (sorted for stability)
ALL_ACTS = sorted(df["concept:name"].unique().tolist())
PAD_TOKEN = "<PAD>"
CATS = [PAD_TOKEN] + ALL_ACTS

max_k = max(train_df["k"].max(), val_df["k"].max(), test_df["k"].max())
cols  = [f"e{i+1}" for i in range(max_k)]

def pad_prefix(p): 
    return p + [PAD_TOKEN] * (max_k - len(p))

X_train_df = pd.DataFrame(train_df["prefix"].apply(pad_prefix).tolist(), columns=cols)
X_val_df   = pd.DataFrame(val_df["prefix"].apply(pad_prefix).tolist(),   columns=cols)
X_test_df  = pd.DataFrame(test_df["prefix"].apply(pad_prefix).tolist(),  columns=cols)

# Force identical categories per column so PAD=0, activities=1.., consistent with LSTM’s “+1” scheme
feat_enc = OrdinalEncoder(categories=[CATS]*max_k)  # no unknowns now
feat_enc.fit(pd.DataFrame([[PAD_TOKEN]*max_k], columns=cols))  # trivial fit; categories already fixed

X_train = feat_enc.transform(X_train_df).astype("float32")
X_val   = feat_enc.transform(X_val_df).astype("float32")
X_test  = feat_enc.transform(X_test_df).astype("float32")

# Labels: fit on ALL next_activity (same policy as LSTM)
label_enc = LabelEncoder().fit(act_df["next_activity"])
y_train = label_enc.transform(train_df["next_activity"])
y_val   = label_enc.transform(val_df["next_activity"])
y_test  = label_enc.transform(test_df["next_activity"])

wandb.config.update({
    "n_classes": int(len(label_enc.classes_)),
    "max_k": int(max_k)
}, allow_val_change=True)

# %%
use_manyclass = False
try:
    from tabpfn_extensions.many_class import ManyClassClassifier
    use_manyclass = True
    log.info("Using ManyClassClassifier wrapper for >10 classes.")
except Exception as e:
    log.warning("tabpfn_extensions.many_class not found (%s). Proceeding with base TabPFNClassifier; ensure n_classes <= 10.", e)

base_clf = TabPFNClassifier(
    device=str(DEVICE),
    ignore_pretraining_limits=True
)

# (Optional) try to set an inference batch size if supported by your version
for key in ("inference_batch_size", "inference_max_batch_size", "batch_size_inference"):
    try:
        base_clf.set_params(**{key: 256})
        log.info("Set %s=256 on TabPFNClassifier", key)
        break
    except Exception:
        pass

clf = (ManyClassClassifier(
          estimator=base_clf,
          alphabet_size=10,
          n_estimators_redundancy=3,
          random_state=SEED,
          verbose=1
       ) if use_manyclass and len(label_enc.classes_) > 10 else base_clf)

# Downsample to keep TabPFN stable/fast on large datasets
SAMPLE_SIZE = 10000  # e.g., 5k–20k; tune as needed
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

# clf.fit(X_train, y_train)

def predict_labels(model, X):
    """Return (pred, proba_or_None)."""
    try:
        proba = model.predict_proba(X)
        return proba.argmax(axis=1), proba
    except Exception:
        return model.predict(X), None

# %%
pred_val, _ = predict_labels(clf, X_val)
acc_val = accuracy_score(y_val, pred_val)
prec_val, rec_val, f1_val, _ = precision_recall_fscore_support(
    y_val, pred_val, average="weighted", zero_division=0
)
wandb.log({
    "val/accuracy": float(acc_val),
    "val/precision": float(prec_val),
    "val/recall": float(rec_val),
    "val/f1": float(f1_val)
})

# %%
pred_test, _ = predict_labels(clf, X_test)

k_vals, counts = [], []
accuracies, precisions, recalls, fscores = [], [], [], []

for k in range(1, max_k + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue
    idx = subset.index.values  # indices into test_df / X_test / y_test order
    yt, yp = y_test[idx], pred_test[idx]

    acc = accuracy_score(yt, yp)
    prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)

    k_vals.append(k); counts.append(len(idx))
    accuracies.append(acc); precisions.append(prec); recalls.append(rec); fscores.append(f1)

# Macro averages across k
avg_accuracy = float(np.mean(accuracies)) if accuracies else float("nan")
avg_precision= float(np.mean(precisions)) if precisions else float("nan")
avg_recall   = float(np.mean(recalls))    if recalls    else float("nan")
avg_f1       = float(np.mean(fscores))    if fscores    else float("nan")

print(f"Average accuracy across all prefixes:  {avg_accuracy:.4f}")
print(f"Average f-score across all prefixes:   {avg_f1:.4f}")
print(f"Average precision across all prefixes: {avg_precision:.4f}")
print(f"Average recall across all prefixes:    {avg_recall:.4f}")

# %%
plot_dir = "/ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/plots/Baselines/TabPFN/ACT"
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
y_pred_all = pred_test
y_true_lbl = [str(s).strip() for s in label_enc.inverse_transform(y_test)]
y_pred_lbl = [str(s).strip() for s in label_enc.inverse_transform(y_pred_all)]
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
    cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=cm_labels)
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
sample = test_df.sample(n=min(5, len(test_df)), random_state=42) if len(test_df) else test_df
table = wandb.Table(columns=["k", "prefix", "gold", "pred", "p_pred", "top5", "top5_p"])
for _, r in sample.iterrows():
    x = feat_enc.transform([pad_prefix(r["prefix"])])
    # Try to get probabilities (ManyClass / TabPFN may or may not support predict_proba consistently)
    try:
        proba = clf.predict_proba(x)[0]
        top_idx = np.argsort(proba)[-5:][::-1]
        pred_id = top_idx[0]
        p_pred  = float(proba[pred_id])
        top5_lbl = label_enc.inverse_transform(top_idx)
        top5_p   = [float(proba[i]) for i in top_idx]
    except Exception:
        y_hat = clf.predict(x)
        pred_id = y_hat[0]; p_pred = float("nan")
        top5_lbl = [label_enc.inverse_transform([pred_id])[0]]
        top5_p   = [float("nan")]

    pred_lbl = label_enc.inverse_transform([pred_id])[0]
    print("Prefix:", " → ".join(r["prefix"]))
    print("Gold:  ", r["next_activity"])
    print(f"Pred:  {pred_lbl} ({p_pred:.3f})")
    print("Top-5:", list(top5_lbl))
    print("-"*60)

    table.add_data(
        r["k"],
        " → ".join(r["prefix"]),
        r["next_activity"],
        pred_lbl,
        p_pred,
        ", ".join([str(x) for x in top5_lbl]),
        ", ".join([f"{p:.3f}" for p in top5_p])
    )
wandb.log({"samples": table})

# %%
model = clf
ts_dir = f"/tmp/tabpfn_ACT_{ts}"
os.makedirs(ts_dir, exist_ok=True)
joblib.dump(clf, f"{ts_dir}/tabpfn_model.pkl")  # ManyClass/TabPFN estimator

artifact = wandb.Artifact(name=f"tabpfn_ACT_artifacts_{ts}", type="model")
artifact.add_dir(ts_dir)
wandb.log_artifact(artifact)

# %%
wandb.finish()