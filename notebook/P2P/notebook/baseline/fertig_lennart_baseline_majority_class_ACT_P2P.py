# %% Majority Baseline — Next Activity (ACT)
# - Rule: always predict the most frequent next activity from the TRAIN set.
# - Metrics: per-k (accuracy, weighted F1/precision/recall) + global VAL/TEST micro accuracy.
# - Logging: W&B curves/tables + confusion matrix; simple sample table using the majority class.
# - Plots: accuracy/F1 vs. prefix length k (saved headlessly to disk).

import os, sys, glob, ctypes
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime

import wandb

from sklearn import metrics

# Data Pipeline
from data import loader
from data.constants import Task

# %%
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %% Config
DATASET = "P2P"

config = {
    # bookkeeping
    "dataset":                  DATASET
}

# %%
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project=f"baseline_majority_ACT_{config['dataset']}",
    entity="privajet-university-of-mannheim",
    name=f"majority_act_{ts}",
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

inv_y = {v: k for k, v in y_word_dict.items()}

# %%
vc = train_df["next_act"].value_counts()
majority_label = vc.idxmax()
p_majority = float(vc.loc[majority_label] / vc.sum())
print(f"Majority next activity (train): {majority_label}  (p={p_majority:.3f})")

# %%
k_vals, accuracies, fscores, precisions, recalls, counts = [], [], [], [], [], []

for i in range(int(max_case_length)):
    subset = test_df[test_df["k"] == i]
    if len(subset) > 0:
        y_true = subset["next_act"].tolist()
        y_pred = [majority_label] * len(y_true)

        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        k_vals.append(i); counts.append(len(y_true))
        accuracies.append(accuracy); fscores.append(fscore); precisions.append(precision); recalls.append(recall)


avg_accuracy = float(np.mean(accuracies)) if accuracies else float("nan")
avg_f1 = float(np.mean(fscores)) if fscores else float("nan")
avg_precision = float(np.mean(precisions)) if precisions else float("nan")
avg_recall = float(np.mean(recalls)) if recalls else float("nan")

print(f"Average accuracy across all prefixes:  {avg_accuracy:.4f}")
print(f"Average f-score across all prefixes:   {avg_f1:.4f}")
print(f"Average precision across all prefixes: {avg_precision:.4f}")
print(f"Average recall across all prefixes:    {avg_recall:.4f}")

# Micro (global) accuracy over all VAL prefixes
y_val_true = val_df["next_act"].tolist()
y_val_pred = [majority_label] * len(y_val_true)
micro_acc = metrics.accuracy_score(y_val_true, y_val_pred)
print(f"[VAL] Micro (global) accuracy: {micro_acc:.4f}")

# Micro (global) accuracy over all TEST prefixes
y_test_true = test_df["next_act"].tolist()
y_test_pred = [majority_label] * len(y_test_true)
micro_acc = metrics.accuracy_score(y_test_true, y_test_pred)
print(f"[TEST] Micro (global) accuracy: {micro_acc:.4f}")

# %% Plots → disk
plot_dir = f"/ceph/lfertig/Thesis/notebook/{config['dataset']}/plots/Baselines/MAJ/ACT"
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

# %% Log per-k curves + macro averages (same keys)
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

# %% Sample predictions (print + W&B table)
sample = test_df.sample(n=min(5, len(test_df)), random_state=42) if len(test_df) else test_df
table = wandb.Table(columns=["k", "prefix", "gold", "pred", "p_pred", "top5", "top5_p"])

for _, r in sample.iterrows():
    prefix_tokens = str(r["prefix"]).split()
    pred = majority_label
    p_pred = p_majority
    top5 = [majority_label]    # majority-only baseline
    top5_p = [p_majority]

    prefix_pretty = " → ".join(prefix_tokens)
    gold = str(r["next_act"])

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
        float(p_pred),
        ", ".join(top5),
        ", ".join([f"{x:.3f}" for x in top5_p])
    )

wandb.log({"samples": table})

# %%
wandb.finish()