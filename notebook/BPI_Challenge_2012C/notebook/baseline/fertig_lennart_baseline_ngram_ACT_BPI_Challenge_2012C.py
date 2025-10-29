# %%
import os, sys, glob, ctypes
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from collections import defaultdict, Counter  # ← missing before

import wandb
from sklearn import metrics

# Data Pipeline
from data import loader
from data.constants import Task

# %%
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %% Config
DATASET = "BPI_Challenge_2012C"

config = {
    # bookkeeping
    "dataset":                  DATASET,
    "n_values":                 [2, 3, 4, 5, 6, 7, 8],  # candidate n for validation
}

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project=f"baseline_ngram_ACT_{config['dataset']}",
    entity="privajet-university-of-mannheim",
    name=f"ngram_act_{ts}",
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

# %%
def fit_ngram(df_part, n: int):
    """Train n-gram model (counts) on df_part."""
    model = defaultdict(Counter)
    for _, r in df_part.iterrows():
        tokens = str(r["prefix"]).split()
        if len(tokens) >= n - 1:
            ctx = tuple(tokens[-(n - 1):])
            model[ctx][r["next_act"]] += 1
    return model

def predict_topk(prefix, model, n: int, k: int, maj_label: str):
    """Return top-k labels + probs from n-gram counts, fallback to majority."""
    tokens = prefix if isinstance(prefix, (list, tuple)) else str(prefix).split()
    if len(tokens) >= n - 1:
        ctx = tuple(tokens[-(n - 1):])
        if ctx in model and len(model[ctx]):
            cnts = model[ctx]
            total = sum(cnts.values()) if sum(cnts.values()) > 0 else 1
            top = cnts.most_common(k)
            labels = [lbl for (lbl, _) in top]
            probs  = [c / total for (_, c) in top]
            return labels, probs
    return [maj_label], [1.0]

def predict_one(prefix, model, n: int, maj_label: str):
    tokens = prefix if isinstance(prefix, (list, tuple)) else str(prefix).split()
    if len(tokens) >= n - 1:
        ctx = tuple(tokens[-(n - 1):])
        if ctx in model and len(model[ctx]):
            return model[ctx].most_common(1)[0][0]
    return maj_label

maj_train = train_df["next_act"].value_counts().idxmax()

# Select n via VAL micro-accuracy
best_n, best_acc = None, -1.0
for n in config["n_values"]:
    model_n = fit_ngram(train_df, n)
    if len(val_df):
        y_val_true = val_df["next_act"].tolist()
        y_val_pred = [predict_one(p, model_n, n, maj_train) for p in val_df["prefix"]]
        acc = metrics.accuracy_score(y_val_true, y_val_pred)
        if acc > best_acc:
            best_acc, best_n = acc, n

# Safe default if VAL is empty or didn’t improve
if best_n is None:
    best_n, best_acc = config["n_values"][0], float("nan")

n_final = int(best_n)
ngram_model = fit_ngram(train_df, n_final)
print(f"Selected n={n_final} via VAL micro-accuracy={best_acc:.4f}")

# %% Per-k loop over actual k values; compute macro averages over k; micro Accuracy
k_vals, accuracies, fscores, precisions, recalls, counts = [], [], [], [], [], []

for i in range(int(max_case_length)):
    subset = test_df[test_df["k"] == i]
    if len(subset):
        y_true = subset["next_act"].tolist()
        y_pred = [predict_one(p, ngram_model, n_final, maj_train) for p in subset["prefix"]]
        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        k_vals.append(i); counts.append(len(y_true))
        accuracies.append(accuracy); fscores.append(fscore); precisions.append(precision); recalls.append(recall)

avg_accuracy = float(np.mean(accuracies)) if accuracies else float("nan")
avg_f1       = float(np.mean(fscores))    if fscores    else float("nan")
avg_precision= float(np.mean(precisions)) if precisions else float("nan")
avg_recall   = float(np.mean(recalls))    if recalls    else float("nan")

print(f"Average accuracy across all prefixes:  {avg_accuracy:.4f}")
print(f"Average f-score across all prefixes:   {avg_f1:.4f}")
print(f"Average precision across all prefixes: {avg_precision:.4f}")
print(f"Average recall across all prefixes:    {avg_recall:.4f}")

# Global VAL/TEST micro-accuracy
y_val_true = val_df["next_act"].tolist()
y_val_pred = [predict_one(p, ngram_model, n_final, maj_train) for p in val_df["prefix"]]
print(f"[VAL]  Micro (global) accuracy: {metrics.accuracy_score(y_val_true, y_val_pred):.4f}")

y_test_true = test_df["next_act"].tolist()
y_test_pred = [predict_one(p, ngram_model, n_final, maj_train) for p in test_df["prefix"]]
print(f"[TEST] Micro (global) accuracy: {metrics.accuracy_score(y_test_true, y_test_pred):.4f}")

# %% Plots → disk
plot_dir = f"/ceph/lfertig/Thesis/notebook/{config['dataset']}/plots/Baselines/NGRAM/ACT"
os.makedirs(plot_dir, exist_ok=True)

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
    "model/ngram_n": n_final,
})

# %% Confusion matrix
y_true_lbl = [str(s).strip() for s in y_test_true]
y_pred_lbl = [str(s).strip() for s in y_test_pred]
cm_labels = sorted(set(y_true_lbl) | set(y_pred_lbl))

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
    cm = metrics.confusion_matrix(y_true_lbl, y_pred_lbl, labels=cm_labels)  # ← fixed
    plt.figure(figsize=(max(6, len(cm_labels)*0.6), max(5, len(cm_labels)*0.5)))
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(ticks=range(len(cm_labels)), labels=cm_labels, rotation=90)
    plt.yticks(ticks=range(len(cm_labels)), labels=cm_labels)
    plt.tight_layout()
    cm_path = os.path.join(plot_dir, f"confusion_matrix_{ts}.png")
    plt.savefig(cm_path, dpi=150); plt.close()
    wandb.log({"cm_image": wandb.Image(cm_path)})

# %% Samples
sample = test_df.sample(n=min(5, len(test_df)), random_state=42) if len(test_df) else test_df
table = wandb.Table(columns=["k", "prefix", "gold", "pred", "p_pred", "top5", "top5_p"])
for _, r in sample.iterrows():
    prefix_str = str(r["prefix"])
    prefix_tokens = prefix_str.split()
    top5, top5_p = predict_topk(prefix_tokens, ngram_model, n_final, 5, maj_train)
    pred = top5[0]; p_pred = float(top5_p[0])
    gold = str(r["next_act"])
    prefix_pretty = " → ".join(prefix_tokens)
    print("Prefix:", prefix_pretty)
    print("Gold:  ", gold)
    print(f"Pred:  {pred} ({p_pred:.3f})")
    print("Top-5:", top5)
    print("-"*60)
    table.add_data(r["k"], prefix_pretty, gold, pred, p_pred,
                   ", ".join(top5), ", ".join([f"{x:.3f}" for x in top5_p]))
wandb.log({"samples": table})

# %%
wandb.finish()