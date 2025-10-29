# %% Zero-shot (ICL-free) Next-Activity prediction with GPT-Neo-1.3B
# - Fully comparable to your LSTM/Transformer/TabPFN/NGRAM/ICL runs
# - Temporal split by case start, per-k metrics, W&B keys, confusion matrix, samples table
import os, sys, glob, ctypes

# Prefer the active conda environment's libstdc++
prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
candidates = glob.glob(os.path.join(prefix, "lib", "libstdc++.so.6*"))
if candidates:
    try:
        mode = getattr(os, "RTLD_GLOBAL", 0)
        ctypes.CDLL(candidates[0], mode=mode)
        # print("Preloaded libstdc++:", candidates[0])  # optional debug
    except OSError as e:
        print("WARN: could not preload libstdc++ from env:", e)

# Headless plotting + avoid torchvision import from HF
os.environ["MPLBACKEND"] = "Agg"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Safe matplotlib import (now uses the preloaded libstdc++)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import random, logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
# Repro + logging
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)
log.info("PyTorch: %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available(): log.info("GPU: %s", torch.cuda.get_device_name(0))

# %%
# W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="gpt-neo-1.3B_ACT_Zero-Shot_P2P",
    entity="privajet-university-of-mannheim",
    name=f"neo_zeroshot_act_{ts}",
    config={"seed": SEED}
)

# %%
# Data
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_p2p.csv.gz", parse_dates=["time:timestamp"])
df = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)

# Build all prefixes (1..n-1)
records = []
for case_id, g in df.groupby("case:concept:name"):
    acts = g["concept:name"].tolist()
    for i in range(1, len(acts)):
        records.append({
            "case_id": case_id,
            "prefix": acts[:i],
            "next_activity": acts[i],
            "k": i
        })
act_df = pd.DataFrame(records)

# Temporal split by case start (identical to your baselines)
case_start = df.groupby("case:concept:name")["time:timestamp"].min().reset_index()
case_start = case_start.sort_values("time:timestamp")
case_ids = case_start["case:concept:name"].tolist()

n_total = len(case_ids)
n_train = int(n_total * 0.8)
n_val   = int(n_train * 0.2)

train_ids = case_ids[: n_train - n_val]
val_ids   = case_ids[n_train - n_val : n_train]
test_ids  = case_ids[n_train : ]

train_df = act_df[act_df["case_id"].isin(train_ids)].reset_index(drop=True)
val_df   = act_df[act_df["case_id"].isin(val_ids)].reset_index(drop=True)
test_df  = act_df[act_df["case_id"].isin(test_ids)].reset_index(drop=True)

print(f"Train prefixes: {len(train_df)} - Validation prefixes: {len(val_df)} - Test prefixes: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# Labels: include all that appear (like your encoder fit on full df)
label_list = sorted(act_df["next_activity"].unique())
labels_for_prompt = " | ".join(label_list)
maxlen = act_df["k"].max()

# %%
# Model (GPU fp16 if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

BASE_MODEL = "EleutherAI/gpt-neo-1.3B"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

config = GPTNeoConfig.from_pretrained(BASE_MODEL)
model  = GPTNeoForCausalLM.from_pretrained(
    BASE_MODEL,
    config=config,
    dtype=DTYPE,
).to(DEVICE).eval()

# %%
# Zero-shot scoring (no demos)
PROMPT_TMPL = (
    "System: Predict the next activity in this business process. "
    "Reply with EXACTLY one label from the list.\n"
    "User: {trace}\n"
    f"Labels: [{labels_for_prompt}]\n"
    "Assistant: "
)

# Pre-tokenize labels once with a leading space to match GPT-Neo tokenization
LABEL_IDS = {lbl: tokenizer(" " + lbl, add_special_tokens=False).input_ids for lbl in label_list}

@torch.no_grad()
def score_labels(prefix_tokens: list[str], length_norm: bool = True):
    """Return (scores np.float32 array of size |labels|, prompt_len) for a single prefix."""
    trace = " → ".join(prefix_tokens)
    prompt = PROMPT_TMPL.format(trace=f"[{trace}]")
    P = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    P_ids = P["input_ids"][0].to(DEVICE)
    # Build a batch [prompt + label_i] for all labels
    rows, lens = [], []
    for lbl in label_list:
        L_ids = torch.tensor(LABEL_IDS[lbl], dtype=torch.long, device=DEVICE)
        rows.append(torch.cat([P_ids, L_ids], dim=0))
        lens.append(len(L_ids))
    pad_id = tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad_id)
    attention_mask = (input_ids != pad_id).long()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (B, T, V)
    cut = P_ids.size(0)
    out = []
    for i, lbl in enumerate(label_list):
        L = lens[i]
        lp = torch.log_softmax(logits[i, cut-1:cut-1+L, :], dim=-1)  # (L, V)
        tgt = torch.tensor(LABEL_IDS[lbl], device=DEVICE)
        ll  = lp.gather(-1, tgt.unsqueeze(-1)).sum()
        if length_norm and L > 0:
            ll = ll / L
        out.append(float(ll))
    return np.array(out, dtype=np.float32), P_ids.size(0)

def predict_topk(prefix_tokens: list[str], k: int = 5):
    scores, _ = score_labels(prefix_tokens, length_norm=True)
    probs = np.exp(scores - scores.max()); probs /= probs.sum() if probs.sum() > 0 else 1.0
    idx = probs.argsort()[-k:][::-1]
    labels_k = [label_list[i] for i in idx]
    probs_k  = [float(probs[i]) for i in idx]
    return labels_k[0], labels_k, probs_k[0], probs_k

# %%
# Per-k evaluation (identical shape to your baselines)
k_vals, counts = [], []
accuracies, precisions, recalls, fscores = [], [], [], []

for k in range(1, maxlen + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty: 
        continue

    y_true = subset["next_activity"].tolist()
    y_pred = [predict_topk(p, k=1)[0] for p in subset["prefix"]]

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    k_vals.append(k); counts.append(len(subset))
    accuracies.append(acc); precisions.append(prec); recalls.append(rec); fscores.append(f1)

# Macro averages across k-bins (do NOT append to curves)
avg_acc = float(np.mean(accuracies)) if accuracies else float("nan")
avg_f1  = float(np.mean(fscores))    if fscores    else float("nan")
avg_p   = float(np.mean(precisions)) if precisions else float("nan")
avg_r   = float(np.mean(recalls))    if recalls    else float("nan")

print(f"Average accuracy across all prefixes:  {avg_acc:.4f}")
print(f"Average f-score across all prefixes:   {avg_f1:.4f}")
print(f"Average precision across all prefixes: {avg_p:.4f}")
print(f"Average recall across all prefixes:    {avg_r:.4f}")

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
})

# Optional: top-k accuracy (kept consistent with your TabPFN logging keys)
def topk_accuracy(y_true, topk_labels_list, k=3):
    hits = sum(y_true[i] in topk_labels_list[i][:k] for i in range(len(y_true)))
    return hits / len(y_true) if len(y_true) else float("nan")

# compute once over the full test set
topk_all = [predict_topk(p, k=5)[1] for p in test_df["prefix"]]
y_all    = test_df["next_activity"].tolist()
wandb.log({
    "metrics/top3_acc": float(topk_accuracy(y_all, topk_all, k=3)),
    "metrics/top5_acc": float(topk_accuracy(y_all, topk_all, k=5))
})

# %%
# Confusion matrix (robust, same style)
y_true_all = [str(s).strip() for s in y_all]
y_pred_all = [predict_topk(p, k=1)[0] for p in test_df["prefix"]]
y_pred_all = [str(s).strip() for s in y_pred_all]
cm_labels  = sorted(set(y_true_all) | set(y_pred_all))

try:
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true_all,
            preds=y_pred_all,
            class_names=cm_labels
        )
    })
except Exception as e:
    log.warning("W&B confusion matrix failed (%s); skipping.", e)

# %%
# Plots → disk (headless)
plot_dir = "/ceph/lfertig/Thesis/notebook/P2P/plots/ZS/ACT"
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
# Samples table (same schema)
sample = test_df.sample(n=min(5, len(test_df)), random_state=SEED) if len(test_df) else test_df
table = wandb.Table(columns=["k","prefix","gold","pred","p_pred","top5","top5_p"])
for _, r in sample.iterrows():
    pred, top5, p_pred, top5_p = predict_topk(r["prefix"], k=5)
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
        float(p_pred),
        ", ".join(top5),
        ", ".join([f"{x:.3f}" for x in top5_p])
    )
wandb.log({"samples": table})

# %%
# Finish
wandb.finish()