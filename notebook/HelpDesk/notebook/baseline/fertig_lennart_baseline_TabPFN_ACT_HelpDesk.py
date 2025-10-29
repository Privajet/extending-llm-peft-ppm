# %% TabPFN — Next Activity (ACT) prediction
# - Input features: fixed-length padded activity IDs from x_word_dict (pre-trunc + pre-pad; PAD=0/UNK=1).
# - Model: TabPFNClassifier (GPU/CPU). Optional ManyClassClassifier wrapper when #classes > 10.
# - Training set optional downsampling to `sample_size` for stability/speed.
# - Metrics: accuracy + weighted precision/recall/F1; per-k curves (accuracy/F1) and micro accuracy on VAL/TEST.
# - Inference: top-k (k=5) helper returns labels + probabilities via predict_proba.
# - Logging: W&B curves/tables, confusion matrix, sample predictions.
# - Artifacts: joblib’d model + x/y dictionaries + metadata saved to /tmp and logged to W&B.

import os, sys, random, logging, glob, ctypes, json, joblib, torch
os.environ["MPLBACKEND"] = "Agg"   # headless matplotlib

# Preload libstdc++ on some HPC stacks (no-op if not needed)
prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
cands = glob.glob(os.path.join(prefix, "lib", "libstdc++.so.6*"))
if cands:
    try:
        mode = getattr(ctypes, "RTLD_GLOBAL", 0)
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

from sklearn import metrics

from tabpfn import TabPFNClassifier
try:
    from tabpfn_extensions.many_class import ManyClassClassifier
except Exception:
    ManyClassClassifier = None

# Data Pipeline
from data import loader
from data.constants import Task

# %% 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# %%
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %% Config
DATASET = "HelpDesk"

config = {
    # bookkeeping
    "dataset":                  DATASET,
    # model scale
    "sample_size":              10000,  # downsample train for speed/stability; set None to disable
}

# %%
config["seed"] = 41
tf.keras.utils.set_random_seed(config["seed"])
tf.config.experimental.enable_op_determinism()
random.seed(config["seed"])
np.random.seed(config["seed"])

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project=f"baseline_tabpfn_ACT_{config['dataset']}",
    entity="privajet-university-of-mannheim",
    name=f"tabpfn_ACT_{ts}",
    config=config,
    resume="never",
    force=True
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
use_manyclass = (ManyClassClassifier is not None) and (len(y_word_dict) > 10)
if use_manyclass:
    log.info("Using ManyClassClassifier wrapper for >10 classes.")
else:
    log.warning("ManyClassClassifier unavailable or not needed (<=10 classes).")

model = TabPFNClassifier(device=str(DEVICE), ignore_pretraining_limits=True)
# Give TabPFN inference batch hints (best-effort)
for key in ("inference_batch_size", "inference_max_batch_size", "batch_size_inference"):
    try:
        model.set_params(**{key: 256}); break
    except Exception:
        pass

model = (ManyClassClassifier(
            estimator=model,
            alphabet_size=min(10, len(y_word_dict)),
            n_estimators_redundancy=3,
            random_state=config["seed"],
            verbose=1
         ) if use_manyclass else model)

# Downsample to keep TabPFN stable/fast on large datasets
SAMPLE_SIZE = config["sample_size"]
if SAMPLE_SIZE and len(X_train) > SAMPLE_SIZE:
    rng = np.random.default_rng(config["seed"])
    sel = rng.choice(len(X_train), size=SAMPLE_SIZE, replace=False)
    X_train_fit = X_train[sel]
    y_train_fit = y_train[sel]
    log.info("Downsampled train from %d to %d for TabPFN", len(X_train), len(X_train_fit))
    wandb.log({"train_downsampled_to": int(len(X_train_fit))})
else:
    X_train_fit, y_train_fit = X_train, y_train

model.fit(X_train_fit, y_train_fit)

# %%
def predict_labels(model, X):
    """Return (pred, proba_or_None)."""
    try:
        proba = model.predict_proba(X)
        return proba.argmax(axis=1), proba
    except Exception:
        return model.predict(X), None
    
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

for i in range(int(max_case_length)):
    test_subset = test_df[test_df["k"] == i]
    if len(test_subset) == 0:
        continue
    X_k, y_k = data_loader.prepare_data_next_activity(test_subset, x_word_dict, y_word_dict, max_case_length)
    y_pred_k, _ = predict_labels(model, X_k)
    acc = metrics.accuracy_score(y_k, y_pred_k)
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(y_k, y_pred_k, average="weighted", zero_division=0)
    k_vals.append(i); counts.append(len(y_k))
    accuracies.append(acc); fscores.append(f1); precisions.append(prec); recalls.append(rec)

avg_accuracy = float(np.mean(accuracies)) if accuracies else float("nan")
avg_f1 = float(np.mean(fscores)) if fscores else float("nan")
avg_precision = float(np.mean(precisions)) if precisions else float("nan")
avg_recall = float(np.mean(recalls)) if recalls else float("nan")

print(f"Average accuracy across all prefixes:  {avg_accuracy:.4f}")
print(f"Average f-score across all prefixes:   {avg_f1:.4f}")
print(f"Average precision across all prefixes: {avg_precision:.4f}")
print(f"Average recall across all prefixes:    {avg_recall:.4f}")

# Micro (global) accuracy over all VAL prefixes
y_pred_val, _ = predict_labels(model, X_val)
micro_acc_val = metrics.accuracy_score(y_val, y_pred_val)
print(f"[VAL]  Micro (global) accuracy: {micro_acc_val:.4f}")

# Micro (global) accuracy over all TEST prefixes
y_pred_all, _ = predict_labels(model, X_test)
micro_acc_test = metrics.accuracy_score(y_test, y_pred_all)
print(f"[TEST] Micro (global) accuracy: {micro_acc_test:.4f}")

# %%
plot_dir = f"/ceph/lfertig/Thesis/notebook/{config['dataset']}/plots/Baselines/TabPFN/ACT"
os.makedirs(plot_dir, exist_ok=True)

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

# %% Robust confusion matrix (avoid KeyError by normalizing strings + union classes)
def _norm(s): return str(s).strip()

y_pred_all_test, _ = predict_labels(model, X_test)
y_true_lbl = [_norm(inv_y[i]) for i in y_test]
y_pred_lbl = [_norm(inv_y[i]) for i in y_pred_all_test]
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

# %% Save TabPFN artifact (joblib), plus metadata
save_dir = f"/tmp/tabpfn_ACT_{ts}"
os.makedirs(save_dir, exist_ok=True)

joblib.dump(model, os.path.join(save_dir, "tabpfn_act_model.pkl"))
with open(os.path.join(save_dir, "metadata.json"), "w") as f:
    json.dump({
        "seed": int(config["seed"]),
        "device": str(DEVICE),
        "maxlen": int(max_case_length),
        "vocab_size": int(vocab_size),
        "num_classes": int(num_output),
        "train_downsampled_to": int(min(len(X_train), config["sample_size"])) if config["sample_size"] else int(len(X_train))
    }, f)
joblib.dump(x_word_dict, os.path.join(save_dir, "x_word_dict.pkl"))
joblib.dump(y_word_dict, os.path.join(save_dir, "y_word_dict.pkl"))

# %% Save weights as artifact
artifact = wandb.Artifact(name=f"tabpfn_ACT_artifacts_{config['dataset']}_{ts}", type="model")
artifact.add_dir(save_dir)
run.log_artifact(artifact)

# %%
wandb.finish()