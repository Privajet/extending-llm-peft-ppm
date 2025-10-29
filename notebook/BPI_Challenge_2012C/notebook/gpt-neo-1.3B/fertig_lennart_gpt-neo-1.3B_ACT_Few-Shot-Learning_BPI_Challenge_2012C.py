# %% Few-shot (ICL) Next-Activity prediction with GPT-Neo-1.3B on a CUDA Linux server
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
from collections import Counter, defaultdict

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

# W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project="gpt-neo-1.3B_ACT_Few-Shot-Learning_BPI_Challenge_2012C",
    entity="privajet-university-of-mannheim",
    name=f"llm_act_icl_{ts}",
)

# %% Data
df = pd.read_csv("/ceph/lfertig/Thesis/data/processed/df_bpi_challenge.csv.gz")
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
labels_for_prompt = " | ".join(label_list)
maxlen = act_df["k"].max()

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

#%%
# Candidate pruning stats from TRAIN
next_by_last = defaultdict(Counter)
for _, r in train_df.iterrows():
    last = r["prefix"][-1]                   # last activity in prefix
    next_by_last[last][r["next_activity"]] += 1

def candidate_labels(prefix, K=12):
    """
    Return up to K most likely next-activity labels, *conditioned on the last event*
    observed in the prefix. This is a cheap, fast pruning step that shrinks the
    label space before scoring with the LLM.

    Inputs
    ------
    prefix : list[str]
        The activity sequence so far (e1, e2, ..., ek).
    K : int
        Max number of candidate labels to return.

    Relies on
    ---------
    - `next_by_last` (a dict: last_activity -> Counter(next_activity frequency))
      which was built from training data earlier in the script.

    Returns
    -------
    list[str] of candidate labels sorted by frequency (most common first).
    """
    last = prefix[-1]
    cands = [a for a, _ in next_by_last[last].most_common(K)]
    if not cands:  # fallback (rare)
        cands = [a for a, _ in Counter(train_df["next_activity"]).most_common(K)]
    return cands

# %% Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_LEN = 512  # match FT's max_seq_length

BASE_MODEL = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="right")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"  # preserve the tail with 'Assistant: '

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    low_cpu_mem_usage=True,
    torch_dtype=DTYPE,
).to(DEVICE).eval()

# Pre-tokenize labels WITHOUT leading space (aligned with FT),
# and we will ensure the prompt ends with exactly one trailing space after 'Assistant:'.
LABEL_IDS = {lbl: tokenizer(lbl, add_special_tokens=False).input_ids for lbl in label_list}

# %% Retrieval for few-shot demos
def seq_str(pfx):  # simple text for TF-IDF; separator won't affect alignment
    """
    Turn a list of activities into a single whitespace-separated string.
    This feeds the TF-IDF vectorizer for retrieval.

    Example: ["A", "B", "C"] -> "A B C"

    Notes:
    - The exact separator isn’t critical for TF-IDF, but spaces are simplest.
    """
    return " ".join(pfx)

train_df["prefix_str"] = train_df["prefix"].apply(seq_str)
tfidf       = TfidfVectorizer().fit(train_df["prefix_str"])
train_tfidf = tfidf.transform(train_df["prefix_str"])

N_SHOTS = 5  # few-shot count (doesn't affect scoring rule alignment)
MAX_DEMO_EVENTS = 10
MAX_QUERY_EVENTS = 20

def retrieve_demos(prefix, n_shots=N_SHOTS):
    """
    Retrieve up to `n_shots` *diverse* few-shot demonstrations from TRAIN that are
    similar to the query `prefix`, using TF-IDF cosine similarity.

    Strategy
    --------
    1) Compute similarity of the query prefix vs. *all* train prefixes.
    2) Take top ~2*n_shots indices (oversample).
    3) Greedily pick examples with *distinct gold labels* to diversify demos.
    4) Truncate each demo prefix to last `MAX_DEMO_EVENTS` to keep prompts short.

    Inputs
    ------
    prefix : list[str]
    n_shots : int

    Relies on
    ---------
    - `tfidf` and `train_tfidf` (fitted on train prefixes)
    - `train_df` with columns ["prefix", "next_activity"]

    Returns
    -------
    demos : list[ (list[str] truncated_prefix, str gold_label) ]
    """
    q    = tfidf.transform([seq_str(prefix)])
    sims = cosine_similarity(q, train_tfidf).ravel()
    topk = np.argsort(sims)[-n_shots*2:][::-1]  # over-sample then dedup by label
    demos, seen_g = [], set()
    for ridx in topk:
        ex = train_df.iloc[ridx]
        g  = ex["next_activity"]
        if g in seen_g:
            continue
        p_demo = ex["prefix"][-MAX_DEMO_EVENTS:]
        demos.append((p_demo, g))
        seen_g.add(g)
        if len(demos) >= n_shots:
            break
    return demos

#%% Pruned prompt + scorer
def make_prompt_with_demos_cands(prefix, n_shots=N_SHOTS, max_q=MAX_QUERY_EVENTS, K=12):
    """
    Build the *text prompt* that includes:
      - up to `n_shots` retrieved demonstrations,
      - a pruned label set (top-K from candidate_labels),
      - the query prefix,
      - and a final 'Answer:' cue.

    The demos are filtered so that their gold labels are within the pruned
    candidate label set (keeps the format consistent across demos & query).

    Format per block:
      Events: A | B | C
      Stats:  k=3 | last=C
      Labels: [L1 | L2 | ...]
      Answer: GOLD

    Query block ends with:
      "Labels: [ ... ]\nAnswer:"

    Inputs
    ------
    prefix : list[str]
    n_shots : int
    max_q   : int  (truncate query prefix to last `max_q` events)
    K       : int  (size of pruned label set)

    Returns
    -------
    prompt : str   (ready for tokenization)
    cands  : list[str]   (the pruned label list to be scored later)
    """
    cands = candidate_labels(prefix, K=K)
    labs_txt = " | ".join(cands)
    # keep only demos whose gold is in cands
    demos = [d for d in retrieve_demos(prefix, n_shots=n_shots) if d[1] in cands]

    def stats(seq):
        return f"k={len(seq)} | last={seq[-1] if seq else '<START>'}"

    blocks = []
    for p, g in demos:
        blocks += [
            f"Events: {' | '.join(p)}\n",
            f"Stats: {stats(p)}\n",
            f"Labels: [{labs_txt}]\n",
            f"Answer: {g}\n\n",
        ]
    q = prefix[-max_q:]
    blocks += [
        f"Events: {' | '.join(q)}\n",
        f"Stats: {stats(q)}\n",
        f"Labels: [{labs_txt}]\n",
        "Answer:"
    ]
    return "".join(blocks), cands

@torch.no_grad()
def predict_icls_pruned(prefix, K=12):
    """
    Few-shot *scoring* of pruned labels using next-token log-probabilities.

    Steps
    -----
    1) Build prompt + pruned candidate list (K labels).
    2) Tokenize prompt *plus a single trailing space*.  (We deliberately do
       *not* add leading spaces to labels; see LABEL_IDS.)
    3) For each candidate label `lbl`, append its token IDs to the prompt IDs,
       pack them into a batch, and run a *single forward pass*.
    4) Sum the log-probabilities of the label tokens at the corresponding positions.
       The label with highest total log-prob wins.

    Inputs
    ------
    prefix : list[str]
    K : int

    Returns
    -------
    best_label : str
    """
    prompt, cands = make_prompt_with_demos_cands(prefix, K=K)
    # score only pruned candidates
    base_ids = tokenizer(prompt + " ", add_special_tokens=False, truncation=True, max_length=MAX_LEN).input_ids
    rows, lens = [], []
    for lbl in cands:
        lid = LABEL_IDS[lbl]
        rows.append(torch.tensor(base_ids + lid, dtype=torch.long))
        lens.append(len(lid))

    pad = tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad).to(DEVICE)
    attn = (input_ids != pad).long()
    logits = model(input_ids=input_ids, attention_mask=attn).logits

    cut = len(base_ids)
    scores = []
    for i, lbl in enumerate(cands):
        L = lens[i]
        lp  = torch.log_softmax(logits[i, cut-1:cut-1+L, :], dim=-1)
        tgt = torch.tensor(LABEL_IDS[lbl], device=attn.device)
        scores.append(float(lp.gather(-1, tgt.unsqueeze(-1)).sum()))
    return cands[int(np.argmax(scores))]

def _softmax(x):
    """
    Numerically-stable softmax over a 1D array (numpy).
    Used to turn unnormalized log-prob scores into a (pseudo) probability
    distribution over candidate labels.
    """
    x = np.array(x, dtype=np.float32); x -= x.max()
    e = np.exp(x); s = e.sum()
    return (e/s) if s > 0 else np.ones_like(x)/len(x)

@torch.no_grad()
def predict_with_topk_pruned(prefix, K=12, topk=5):
    """
    Like `predict_icls_pruned`, but returns:
      - the best label,
      - the top-k labels,
      - the best label's probability,
      - and the top-k probabilities.

    Steps
    -----
    1) Build prompt + pruned candidates.
    2) Batch-score log-probs for each candidate label.
    3) Apply softmax over scores for display.
    4) Extract top-k.

    Returns
    -------
    best : str
    top_lbl : list[str] (top-k labels in descending prob order)
    p_best : float
    top_p : list[float] (top-k probabilities, same order as top_lbl)
    """
    prompt, cands = make_prompt_with_demos_cands(prefix, K=K)
    base_ids = tokenizer(prompt + " ", add_special_tokens=False, truncation=True, max_length=MAX_LEN).input_ids
    rows, lens = [], []
    for lbl in cands:
        lid = LABEL_IDS[lbl]
        rows.append(torch.tensor(base_ids + lid, dtype=torch.long))
        lens.append(len(lid))
    pad = tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad).to(DEVICE)
    attn = (input_ids != pad).long()
    logits = model(input_ids=input_ids, attention_mask=attn).logits
    cut = len(base_ids)
    scores = []
    for i, lbl in enumerate(cands):
        L = lens[i]
        lp  = torch.log_softmax(logits[i, cut-1:cut-1+L, :], dim=-1)
        tgt = torch.tensor(LABEL_IDS[lbl], device=attn.device)
        scores.append(float(lp.gather(-1, tgt.unsqueeze(-1)).sum()))
    scores = np.array(scores, dtype=np.float32)
    probs = _softmax(scores)
    top_idx = np.argsort(probs)[-min(topk, len(cands)):][::-1]
    top_lbl = [cands[i] for i in top_idx]
    top_p   = [float(probs[i]) for i in top_idx]
    best, p_best = top_lbl[0], top_p[0]
    return best, top_lbl, p_best, top_p

# %% Per-k evaluation (uses pruned predictor)
K_PRUNE = 12  # try {8,12,16,24}
k_vals, counts = [], []
accuracies, precisions, recalls, fscores = [], [], [], []

for k in range(1, maxlen + 1):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue
    y_true = subset["next_activity"].tolist()
    y_pred = [predict_icls_pruned(p, K=K_PRUNE) for p in subset["prefix"]]

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

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
    "prune/K": K_PRUNE,
})

# %% Confusion matrix (with pruned preds)
y_true_all = test_df["next_activity"].tolist()
y_pred_all = [predict_icls_pruned(p, K=K_PRUNE) for p in test_df["prefix"]]
cm_labels  = sorted(set(map(str, y_true_all)) | set(map(str, y_pred_all)))

try:
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=[str(x).strip() for x in y_true_all],
            preds=[str(x).strip() for x in y_pred_all],
            class_names=cm_labels
        )
    })
except Exception as e:
    log.warning("W&B CM failed: %s (skipping).", e)

# %% Plots → disk
plot_dir = "/ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/plots/FS/ACT"
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

# %% Sample table (pruned top-k)
sample = test_df.sample(n=min(5, len(test_df)), random_state=SEED) if len(test_df) else test_df
table = wandb.Table(columns=["k","prefix","gold","pred","p_pred","top5","top5_p"])
for _, r in sample.iterrows():
    pred, top5, p_pred, top5_p = predict_with_topk_pruned(r["prefix"], K=K_PRUNE, topk=5)
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