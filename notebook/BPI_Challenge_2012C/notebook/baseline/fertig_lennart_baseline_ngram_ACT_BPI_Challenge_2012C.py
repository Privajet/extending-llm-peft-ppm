# %%
import os
os.environ["MPLBACKEND"] = "Agg"  # headless matplotlib

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from collections import defaultdict, Counter

import wandb
from sklearn import metrics

# %%
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

config = {
    "baseline": "ngram",
    "n_values": [2, 3, 4, 5, 6, 7, 8],  # candidate n for validation
}
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="baseline_ngram_ACT_BPI_Challenge_2012C",
    entity="privajet-university-of-mannheim",
    name=f"ngram_act_{ts}",
    config=config,
    resume="never",
    force=True
)

# %%
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_bpi_challenge.csv.gz")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)

# Build prefixes (1..n-1) for Next-Activity prediction
rows = []
for case_id, g in df.groupby("case:concept:name"):
    acts = g["concept:name"].tolist()
    for i in range(1, len(acts)):
        rows.append({
            "case_id": case_id,
            "prefix": acts[:i],
            "next_activity": acts[i],
            "k": i
        })
n_gram_act_prefix_df = pd.DataFrame(rows)

case_start = df.groupby("case:concept:name")["time:timestamp"].min().reset_index()
case_start = case_start.sort_values("time:timestamp")
case_ids   = case_start["case:concept:name"].tolist()

n_total = len(case_ids)
n_train = int(n_total * 0.8)
n_val   = int(n_train * 0.2)

train_ids = case_ids[:n_train - n_val]
val_ids   = case_ids[n_train - n_val:n_train]
test_ids  = case_ids[n_train:]

train_df = n_gram_act_prefix_df[n_gram_act_prefix_df["case_id"].isin(train_ids)].reset_index(drop=True)
val_df   = n_gram_act_prefix_df[n_gram_act_prefix_df["case_id"].isin(val_ids)].reset_index(drop=True)
test_df  = n_gram_act_prefix_df[n_gram_act_prefix_df["case_id"].isin(test_ids)].reset_index(drop=True)

print(f"Train prefixes: {len(train_df)} - Val prefixes: {len(val_df)} - Test prefixes: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# %%
def fit_ngram(df_part, n: int):
    """Train n-gram model (counts) on df_part."""
    model = defaultdict(Counter)
    for _, r in df_part.iterrows():
        p = r["prefix"]
        if len(p) >= n - 1:
            ctx = tuple(p[-(n - 1):])
            model[ctx][r["next_activity"]] += 1
    return model

def predict_one(prefix, model, n: int, maj_label: str):
    """Top-1 prediction for a single prefix with fallback to majority label."""
    if len(prefix) >= n - 1:
        ctx = tuple(prefix[-(n - 1):])
        if ctx in model and len(model[ctx]):
            return model[ctx].most_common(1)[0][0]
    return maj_label

def predict_topk(prefix, model, n: int, k: int, maj_label: str):
    """Top-k labels (by counts) for optional top-k metrics / display."""
    if len(prefix) >= n - 1:
        ctx = tuple(prefix[-(n - 1):])
        if ctx in model and len(model[ctx]):
            return [lbl for (lbl, _) in model[ctx].most_common(k)]
    # fallback: pad with majority to at least one label
    return [maj_label]

def eval_per_k(df_part, model, n: int, maj_label: str):
    """Per-k classification metrics (macro over k like LSTM/PT scripts)."""
    k_vals, accuracies, fscores, precisions, recalls, counts = [], [], [], [], [], []
    maxlen = df_part["k"].max() if len(df_part) else 0
    for k in range(1, maxlen + 1):
        subset = df_part[df_part["k"] == k]
        if subset.empty:
            continue
        y_true = subset["next_activity"].tolist()
        y_pred = [predict_one(p, model, n, maj_label) for p in subset["prefix"]]

        acc = metrics.accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = metrics.precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        k_vals.append(k); counts.append(len(subset))
        accuracies.append(acc); fscores.append(f1); precisions.append(prec); recalls.append(rec)

    # macro over k (no synthetic point in curves)
    avg_acc = float(np.mean(accuracies)) if accuracies else float("nan")
    avg_f1  = float(np.mean(fscores))    if fscores    else float("nan")
    avg_p   = float(np.mean(precisions)) if precisions else float("nan")
    avg_r   = float(np.mean(recalls))    if recalls    else float("nan")
    return k_vals, counts, accuracies, fscores, precisions, recalls, avg_acc, avg_f1, avg_p, avg_r

# %%
maj_train = train_df["next_activity"].value_counts().idxmax()
best = {"n": None, "avg_f1": -1.0, "avg_acc": -1.0}

for n in config["n_values"]:
    model_val = fit_ngram(train_df, n)
    _, _, accs, f1s, _, _, avg_acc, avg_f1, _, _ = eval_per_k(val_df, model_val, n, maj_train)
    print(f"VAL n={n:>2}: avg_acc={avg_acc:.4f}  avg_f1={avg_f1:.4f}")
    if (avg_f1 > best["avg_f1"]) or (np.isclose(avg_f1, best["avg_f1"]) and avg_acc > best["avg_acc"]):
        best.update({"n": n, "avg_f1": avg_f1, "avg_acc": avg_acc})

wandb.config.update({"n_selected": int(best["n"])}, allow_val_change=True)
print(f"Selected n={best['n']} (val avg_f1={best['avg_f1']:.4f}, avg_acc={best['avg_acc']:.4f})")

# %%
n_final = int(best["n"])
ngram_model = fit_ngram(train_df, n_final)

k_vals, counts, accuracies, fscores, precisions, recalls, avg_acc, avg_f1, avg_p, avg_r = \
    eval_per_k(test_df, ngram_model, n_final, maj_train)

print(f"Average accuracy across all prefixes:  {avg_acc:.4f}")
print(f"Average f-score across all prefixes:   {avg_f1:.4f}")
print(f"Average precision across all prefixes: {avg_p:.4f}")
print(f"Average recall across all prefixes:    {avg_r:.4f}")

# %%
plot_dir = "/ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/plots/Baselines/NGRAM/ACT"
os.makedirs(plot_dir, exist_ok=True)

if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, accuracies, marker='o', label='Accuracy')
    plt.title(f'Accuracy vs. Prefix Length (k), n={n_final}'); plt.xlabel('Prefix Length (k)'); plt.ylabel('Accuracy')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"acc_vs_k_n{n_final}_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, fscores, marker='o', label='F1 (weighted)')
    plt.title(f'F1 vs. Prefix Length (k), n={n_final}'); plt.xlabel('Prefix Length (k)'); plt.ylabel('F1 (weighted)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"f1_vs_k_n{n_final}_{ts}.png"), dpi=150); plt.close()

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
    "config/n_final": n_final
})

# %%
y_true_all = test_df["next_activity"].tolist()
y_pred_all = [predict_one(p, ngram_model, n_final, maj_train) for p in test_df["prefix"]]
# Normalize to strings + union of classes to prevent KeyErrors in W&B
y_true_str = [str(s).strip() for s in y_true_all]
y_pred_str = [str(s).strip() for s in y_pred_all]
cm_classes = sorted(set(y_true_str) | set(y_pred_str))

try:
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=y_true_str, preds=y_pred_str, class_names=cm_classes
        )
    })
except Exception as e:
    print("W&B confusion_matrix failed; skipping. Error:", e)

# %%
def topk_acc(df_part, model, n, k, maj_label):
    hits = 0
    for _, r in df_part.iterrows():
        topk = predict_topk(r["prefix"], model, n, k, maj_label)
        hits += (r["next_activity"] in topk)
    return hits / len(df_part) if len(df_part) else float("nan")

top3 = topk_acc(test_df, ngram_model, n_final, 3, maj_train)
top5 = topk_acc(test_df, ngram_model, n_final, 5, maj_train)
wandb.log({"metrics/top3_acc": float(top3), "metrics/top5_acc": float(top5)})

# %%
def topk_with_probs(prefix, model, n, k, maj_label):
    if len(prefix) >= n-1:
        ctx = tuple(prefix[-(n-1):])
        if ctx in model and len(model[ctx]):
            cnts = model[ctx]
            total = sum(cnts.values())
            top = cnts.most_common(k)
            labels = [lbl for lbl, _ in top]
            probs  = [c/total for _, c in top]
            return labels, probs
    # fallback: majority only
    return [maj_label], [1.0]

sample = test_df.sample(n=min(5, len(test_df)), random_state=42) if len(test_df) else test_df
table = wandb.Table(columns=["k", "prefix", "gold", "pred", "p_pred", "top5", "top5_p"])

for _, r in sample.iterrows():
    top5_lbl, top5_p = topk_with_probs(r["prefix"], ngram_model, n_final, 5, maj_train)
    pred = top5_lbl[0]; p_pred = float(top5_p[0])
    print("Prefix:", " → ".join(r["prefix"]))
    print("Gold:  ", r["next_activity"])
    print(f"Pred:  {pred} ({p_pred:.3f})")
    print("Top-5:", top5_lbl)
    print("-"*60)

    table.add_data(
        r["k"],
        " → ".join(r["prefix"]),
        r["next_activity"],
        pred,
        p_pred,
        ", ".join(top5_lbl),
        ", ".join([f"{p:.3f}" for p in top5_p])
    )

wandb.log({"samples": table})

# %%
# Convert defaultdict(Counter) to a JSON-serializable dict
serializable_model = {str(k): dict(v) for k, v in ngram_model.items()}
model_dir = f"/tmp/ngram_act_{ts}"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"ngram_n{n_final}.json")
with open(model_path, "w") as f:
    json.dump({
        "n": n_final,
        "majority_label_train": maj_train,
        "model": serializable_model
    }, f)

artifact = wandb.Artifact(name=f"ngram_act_artifacts_{ts}", type="model")
artifact.add_file(model_path)
run.log_artifact(artifact)

# %%
wandb.finish()