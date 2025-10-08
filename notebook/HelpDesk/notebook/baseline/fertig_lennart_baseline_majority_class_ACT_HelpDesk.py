# %%
import os
os.environ["MPLBACKEND"] = "Agg"  # headless matplotlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime

import wandb
from sklearn import metrics

# %%
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

config = {
    "baseline": "majority_class",
}
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_majority_ACT_HelpDesk",
    entity="privajet-university-of-mannheim",
    name=f"majority_act_{ts}",
    config=config,
    resume="never",
    force=True
)

# %%
train_df = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_train.csv")
val_df   = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_val.csv")
test_df  = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_test.csv")

for d in (train_df, val_df, test_df):
    d.rename(columns={"next_act": "next_activity"}, inplace=True)

print(f"Train prefixes: {len(train_df)} - Validation prefixes: {len(val_df)} - Test prefixes: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# %%
majority_label = train_df["next_activity"].value_counts().idxmax()
print(f"Majority next activity (train): {majority_label}")

# %%
k_vals, accuracies, fscores, precisions, recalls, counts = [], [], [], [], [], []
maxlen = test_df["k"].max() if len(test_df) else 0

for k in sorted(test_df["k"].astype(int).unique()):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue

    y_true = subset["next_activity"].tolist()
    y_pred = [majority_label] * len(y_true)

    acc = metrics.accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    k_vals.append(k); counts.append(len(subset))
    accuracies.append(acc); fscores.append(f1); precisions.append(prec); recalls.append(rec)

# Macro averages across k-bins
avg_acc = float(np.mean(accuracies)) if accuracies else float("nan")
avg_f1  = float(np.mean(fscores))    if fscores    else float("nan")
avg_p   = float(np.mean(precisions)) if precisions else float("nan")
avg_r   = float(np.mean(recalls))    if recalls    else float("nan")

print(f"Average accuracy across all prefixes:  {avg_acc:.4f}")
print(f"Average f-score across all prefixes:   {avg_f1:.4f}")
print(f"Average precision across all prefixes: {avg_p:.4f}")
print(f"Average recall across all prefixes:    {avg_r:.4f}")

# %%
plot_dir = "/ceph/lfertig/Thesis/notebook/HelpDesk/plots/Baselines/MAJ/ACT"
os.makedirs(plot_dir, exist_ok=True)

if len(k_vals):
    plt.figure(figsize=(8, 5))
    plt.plot(k_vals, accuracies, marker='o', label='Accuracy')
    plt.title('Accuracy vs. Prefix Length (k)')
    plt.xlabel('Prefix Length (k)'); plt.ylabel('Accuracy')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"acc_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(k_vals, fscores, marker='o', label='F1 Score')
    plt.title('F1 Score vs. Prefix Length (k)')
    plt.xlabel('Prefix Length (k)'); plt.ylabel('F1 (weighted)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"f1_vs_k_{ts}.png"), dpi=150); plt.close()

print(f"Saved plots to: {plot_dir}")

# %%
wandb.log({
    "curves/k": k_vals,
    "curves/accuracy": accuracies,
    "curves/f1": fscores,
    "curves/precision": precisions,
    "curves/recall": recalls,
    "curves/counts": counts,
    "metrics/avg_accuracy": avg_acc,
    "metrics/avg_f1": avg_f1,
    "metrics/avg_precision": avg_p,
    "metrics/avg_recall": avg_r,
})

# %%
y_true_all = test_df["next_activity"].tolist()
y_pred_all = [majority_label] * len(test_df)

y_true_str = [str(s).strip() for s in y_true_all]
y_pred_str = [str(s).strip() for s in y_pred_all]
cm_classes = sorted(set(y_true_str) | set(y_pred_str))

try:
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true_str,
            preds=y_pred_str,
            class_names=cm_classes
        )
    })
except Exception as e:
    print("W&B confusion_matrix failed; skipping. Error:", e)

# %%
# Global train distribution for 'probabilities'
train_freq = train_df["next_activity"].value_counts()
train_total = len(train_df)
p_majority = float(train_freq.get(majority_label, 0) / train_total) if train_total else float("nan")
top5_labels = train_freq.index[:5].tolist()
top5_probs  = (train_freq.iloc[:5] / train_total).astype(float).tolist() if train_total else []

# Pad to 5 if classes < 5 (optional)
while len(top5_labels) < 5:
    top5_labels.append("")
    top5_probs.append(0.0)

sample = test_df.sample(n=min(5, len(test_df)), random_state=42) if len(test_df) else test_df
table = wandb.Table(columns=["k", "prefix", "gold", "pred", "p_pred", "top5", "top5_p"])

for _, r in sample.iterrows():
    pred = majority_label
    print("Prefix:", r["prefix"])
    print("Gold:  ", r["next_activity"])
    print(f"Pred:  {pred} ({p_majority:.3f})")
    print("Top-5:", top5_labels)
    print("-"*60)
    table.add_data(
        r["k"],
        " → ".join(r["prefix"]),
        r["next_activity"],
        pred,
        p_majority,
        ", ".join(top5_labels),
        ", ".join([f"{p:.3f}" for p in top5_probs])
    )

wandb.log({"samples": table})

# %%
run.finish()