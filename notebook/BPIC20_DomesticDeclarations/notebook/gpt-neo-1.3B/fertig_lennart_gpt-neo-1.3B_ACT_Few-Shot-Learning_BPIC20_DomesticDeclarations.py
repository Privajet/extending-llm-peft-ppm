# %%
# Few-shot (ICL) Next-Activity prediction with GPT-Neo-1.3B on a CUDA Linux server
import os, sys, glob, ctypes

# Prefer the active conda env's libstdc++.so.6 (has the required CXXABI_1.3.15+)
prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
lib_candidates = sorted(glob.glob(os.path.join(prefix, "lib", "libstdc++.so.6*")))
if lib_candidates:
    try:
        # RTLD_GLOBAL makes symbols visible to later-loaded extension modules
        ctypes.CDLL(lib_candidates[0], mode=getattr(os, "RTLD_GLOBAL", 0))
    except OSError as e:
        print(f"WARN: could not preload {lib_candidates[0]}: {e}")

# Headless plotting + avoid torchvision usage in HF
os.environ["MPLBACKEND"] = "Agg"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Standard imports
import random
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")  # double-safety; respects MPLBACKEND=Agg
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

log.info("PyTorch: %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available():
    log.info("GPU: %s", torch.cuda.get_device_name(0))

# W&B login
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="gpt-neo-1.3B_ACT_Few-Shot-Learning_BPIC20_DomesticDeclarations",
    entity="privajet-university-of-mannheim",
    name=f"llm_act_icl_{ts}",
)

# %%
# Data
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_bpic20_domestic.csv.gz")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)
log.info("Data shape: %s", df.shape)
print(df.head(10))

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
print("Total prefix samples:", len(act_df))

label_list = sorted(act_df["next_activity"].unique())
label2id   = {lbl:i for i,lbl in enumerate(label_list)}
id2label   = {i:lbl for lbl,i in label2id.items()}
maxlen     = act_df["k"].max()

# Temporal split by case start (identical to LSTM)
case_start = (
    df.groupby("case:concept:name")["time:timestamp"]
      .min().reset_index().sort_values("time:timestamp")
)
valid_cases = set(act_df["case_id"])
case_ids = [c for c in case_start["case:concept:name"] if c in valid_cases]

n_total = len(case_ids)
n_train = int(n_total * 0.8)
n_val   = int(n_train * 0.2)

train_ids = case_ids[: n_train - n_val]
val_ids   = case_ids[n_train - n_val : n_train]
test_ids  = case_ids[n_train : ]

train_df = act_df[act_df["case_id"].isin(train_ids)].reset_index(drop=True)
val_df   = act_df[act_df["case_id"].isin(val_ids)  ].reset_index(drop=True)
test_df  = act_df[act_df["case_id"].isin(test_ids) ].reset_index(drop=True)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# %%
# Model
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
    dtype=DTYPE,  # ok on newer transformers; older versions accept torch_dtype (deprecated)
).to(DEVICE).eval()

# %%
# Retrieval for few-shot demos
def seq_str(pfx): return " ".join(pfx)
train_df["prefix_str"] = train_df["prefix"].apply(seq_str)
tfidf       = TfidfVectorizer().fit(train_df["prefix_str"])
train_tfidf = tfidf.transform(train_df["prefix_str"])

CFG = dict(
    n_shots=5,
    max_demo_events=10,
    max_query_events=20,
    max_prompt_tokens=1024,    # < 2048 for GPT-Neo
    length_norm=True,          # divide LL by #label tokens
    exclude_seen=False,        # IMPORTANT: all labels allowed (match LSTM)
    calibration=True,          # contextual calibration
    permutations=3,            # label-order ensembling
    seed=SEED,
)
wandb.config.update(CFG)

LABEL_IDS = {lbl: tokenizer(" " + lbl, add_special_tokens=False).input_ids for lbl in label_list}
ASSIST = "Assistant:"

def _labels_text(labels): return " | ".join(labels)

def retrieve_demos(prefix, n_shots=CFG["n_shots"]):
    q    = tfidf.transform([seq_str(prefix)])
    sims = cosine_similarity(q, train_tfidf).ravel()
    topk = np.argsort(sims)[-n_shots*2:][::-1]  # over-sample then dedup
    demos, seen_g = [], set()
    for ridx in topk:
        ex = train_df.iloc[ridx]
        g  = ex["next_activity"]
        if g in seen_g:
            continue
        p_demo = ex["prefix"][-CFG["max_demo_events"]:]
        demos.append((p_demo, g))
        seen_g.add(g)
        if len(demos) >= n_shots:
            break
    return demos

def make_prompt(demos, query_prefix, labels_block):
    lines = ["Examples:"]
    for i, (p,g) in enumerate(demos, start=1):
        lines.append(f"{i}) [{' → '.join(p)}] → {g}")
    lines.append("")
    lines.append(f"Labels: [{_labels_text(labels_block)}]")
    short_q = query_prefix[-CFG["max_query_events"]:]
    lines.append(f"Query: [{' → '.join(short_q)}] →")
    lines.append(ASSIST)
    prompt = "\n".join(lines)
    ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=CFG["max_prompt_tokens"]).input_ids
    return ids

@torch.no_grad()
def score_labels(base_ids, label_ids_list):
    rows, lens = [], []
    for lid in label_ids_list:
        rows.append(torch.tensor(base_ids + lid, dtype=torch.long))
        lens.append(len(lid))
    pad = tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad).to(DEVICE)
    attn = (input_ids != pad).long()
    logits = model(input_ids=input_ids, attention_mask=attn).logits
    cut = len(base_ids)
    scores = []
    for i, lid in enumerate(label_ids_list):
        L = lens[i]
        lp = torch.log_softmax(logits[i, cut-1:cut-1+L, :], dim=-1)
        tgt = torch.tensor(lid, device=lp.device)
        ll  = float(lp.gather(-1, tgt.unsqueeze(-1)).sum())
        if CFG["length_norm"] and L > 0:
            ll = ll / L
        scores.append(ll)
    return np.array(scores, dtype=np.float32)

def calibration_vector(labels_order):
    if not CFG["calibration"]:
        return np.zeros(len(labels_order), dtype=np.float32)
    neutral = (
        "Examples:\n\n"
        f"Labels: [{_labels_text(labels_order)}]\n"
        "Query: [] →\n"
        f"{ASSIST}"
    )
    base_ids = tokenizer(neutral, add_special_tokens=False, truncation=True, max_length=CFG["max_prompt_tokens"]).input_ids
    label_ids_list = [LABEL_IDS[l] for l in labels_order]
    return score_labels(base_ids, label_ids_list)

def predict_next_activity_few_shot(prefix):
    candidates = label_list[:]  # all labels (match LSTM)
    demos = retrieve_demos(prefix, n_shots=CFG["n_shots"])
    agg = np.zeros(len(candidates), dtype=np.float32)
    for _ in range(CFG["permutations"]):
        labels_perm = candidates[:]
        random.shuffle(labels_perm)
        base_ids = make_prompt(demos, prefix, labels_perm)
        label_ids_list = [LABEL_IDS[l] for l in labels_perm]
        s = score_labels(base_ids, label_ids_list)
        c = calibration_vector(labels_perm)
        s_adj = s - c
        for i, lbl in enumerate(labels_perm):
            agg[candidates.index(lbl)] += s_adj[i]
    return candidates[int(np.argmax(agg))], agg  # return raw scores for top-k

def _softmax(x):
    x = np.array(x, dtype=np.float32)
    x -= np.max(x)
    e = np.exp(x); s = e.sum()
    return (e / s) if s > 0 else np.ones_like(x)/len(x)

def predict_with_topk(prefix, topk=5):
    pred, scores = predict_next_activity_few_shot(prefix)
    probs = _softmax(scores)
    top_idx = np.argsort(probs)[-topk:][::-1]
    top_lbl = [label_list[i] for i in top_idx]
    top_p   = [float(probs[i]) for i in top_idx]
    p_pred  = float(probs[label_list.index(pred)])
    return pred, top_lbl, p_pred, top_p

# %%
# Per-k evaluation
k_vals, counts = [], []
accuracies, precisions, recalls, fscores = [], [], [], []

for k in range(1, maxlen + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue
    y_true = subset["next_activity"].tolist()
    y_pred = [predict_next_activity_few_shot(p)[0] for p in subset["prefix"]]

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    k_vals.append(k); counts.append(len(subset))
    accuracies.append(acc); precisions.append(prec); recalls.append(rec); fscores.append(f1)

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

# %%
# Confusion matrix (same logging style)
y_true_all = test_df["next_activity"].tolist()
y_pred_all = [predict_next_activity_few_shot(p)[0] for p in test_df["prefix"]]
cm_labels  = sorted(set(map(str, y_true_all)) | set(map(str, y_pred_all)))

wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=[str(x).strip() for x in y_true_all],
        preds=[str(x).strip() for x in y_pred_all],
        class_names=cm_labels
    )
})

# %%
# Plots → disk
plot_dir = "/ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/plots/ICL/ACT"
os.makedirs(plot_dir, exist_ok=True)

if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, accuracies, marker="o", label="Accuracy")
    plt.title("Accuracy vs. Prefix Length (k)"); plt.xlabel("Prefix Length (k)"); plt.ylabel("Accuracy")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"acc_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, fscores, marker="o", label="F1 (weighted)")
    plt.title("F1 vs. Prefix Length (k)"); plt.xlabel("Prefix Length (k)"); plt.ylabel("F1 (weighted)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"f1_vs_k_{ts}.png"), dpi=150); plt.close()

print(f"Saved plots to: {plot_dir}")

# %%
sample = test_df.sample(n=min(5, len(test_df)), random_state=SEED) if len(test_df) else test_df
table = wandb.Table(columns=["k","prefix","gold","pred","p_pred","top5","top5_p"])
for _, r in sample.iterrows():
    pred, top5, p_pred, top5_p = predict_with_topk(r["prefix"], topk=5)
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
wandb.finish()