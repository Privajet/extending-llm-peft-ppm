# %% Few-shot (ICL) Next-Activity (ACT) with GPT-Neo-1.3B
# - strict label scoring via log-likelihood of [prompt + label (+EOS)]
# - TF-IDF retrieval for few-shot demos (MMR diversified), temporal split (no leakage)
# - candidate pruning via TRAIN bigrams (top-K), optional soft bigram boost / hard filter
# - length normalization, temperature, class-prior calibration (from TRAIN)
# - tiny validation sweep on VAL for tau/prior_alpha/K_prune/n_shots
# - per-k curves + top-k metrics, W&B logging (prompt + config)
# - speed: cache tokenized prompt IDs; prebuild label tensors on device

import os, sys, glob, ctypes, random, logging
os.environ["MPLBACKEND"] = "Agg"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Preload libstdc++ on some HPC stacks (no-op if not needed)
prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
cands = glob.glob(os.path.join(prefix, "lib", "libstdc++.so.6*"))
if cands:
    try:
        mode = getattr(os, "RTLD_GLOBAL", 0)
        ctypes.CDLL(cands[0], mode=mode)
    except OSError:
        pass

import numpy as np
import pandas as pd
import torch
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter, defaultdict, OrderedDict

# %% 
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %%
DATASET = "HelpDesk"

config = {
    # bookkeeping
    "dataset":                  DATASET,
    "plots_dir":                f"/ceph/lfertig/Thesis/notebook/{DATASET}/plots/gpt-neo-1.3B/FS/ACT"
}

FS_CFG = {
    # model / runtime
    "family":                   "neo",
    "model_name":               "EleutherAI/gpt-neo-1.3B",
    "dtype":                    "fp16",                                 # "fp32" if CPU-only
    "device":                   "auto",
    # prompt & context
    "n_shots":                  5,                                      # grid: [3,5]
    "ctx_events":               12,                                     # query-tail events (aligned name with ZS; used in prompt)
    "max_demo_events":          10,                                     # demo prefix truncation (grid: [6,8,10])
    "event_sep":                " → ",
    "prompt_tmpl_demo":         (
                                "Trace: {trace}\n"
                                "Choose EXACTLY ONE label from the list below and output ONLY that label.\n"
                                "Labels:\n{labels}\n"
                                "Answer: {gold}\n\n"
                                ),
    "prompt_tmpl_query":        (
                                "Trace: {trace}\n"
                                "Choose EXACTLY ONE label from the list below and output ONLY that label.\n"
                                "Labels:\n{labels}\n"
                                "Answer:"
                                ),
    "add_eos_after_label":      True,
    # scoring
    "length_norm":              True,
    "temperature":              0.7,                                    # tune on val
    "use_class_prior":          False,
    "prior_alpha":              0.00,                                   # tune on val
    "bigram_weight":            1.2,                                    # for soft bigram boost (tune on val)
    # transition knowledge
    "use_bigram_boost":         True,                                   # soft bump for labels seen after last event in TRAIN
    "bigram_boost":             0.30,
    "use_bigram_filter":        True,                                   # hard filter to only labels seen after last event
    # pruning
    "K_prune":                  8,                                      # grid: [8,12,16]
    "no_self_loop_if_unseen":   True,                                   # if last event unseen, allow self-loop (otherwise it would be pruned out)
    # validation sweep (tiny)
    "do_val_tune":              False,
    "grid_taus":                [0.6, 0.7, 0.8],
    "grid_alphas":              [0.0, 0.1],
    "grid_K":                   [6, 8, 12],
    "grid_shots":               [3, 5],
    "ctx_events_grid":          [8, 12, 16],
    # evaluation
    "topk":                     [1, 3, 5],
}

# %%
config["seed"] = 41
random.seed(config["seed"]);
np.random.seed(config["seed"]); 
torch.manual_seed(config["seed"])
if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(config["seed"])

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)
log.info("PyTorch: %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available(): log.info("GPU: %s", torch.cuda.get_device_name(0))

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run = wandb.init(
    project=f"gpt-neo-1.3B_ACT_FewShot_{config['dataset']}",
    entity="privajet-university-of-mannheim",
    name=f"neo_fs_act_{ts}",
    config=config,
    resume="never",
    force=True
)

# %% Data
train_df = pd.read_csv(f"/ceph/lfertig/Thesis/data/{config['dataset']}/processed/next_activity_train.csv")
val_df   = pd.read_csv(f"/ceph/lfertig/Thesis/data/{config['dataset']}/processed/next_activity_val.csv")
test_df  = pd.read_csv(f"/ceph/lfertig/Thesis/data/{config['dataset']}/processed/next_activity_test.csv")

for d in (train_df, val_df, test_df):
    d.rename(columns={"next_act": "next_activity"}, inplace=True)
    d["prefix"] = d["prefix"].astype(str).str.split() # convert space-separated strings to lists

print(f"Train prefixes: {len(train_df)} - Validation prefixes: {len(val_df)} - Test prefixes: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# labels: Union of all labels that appear and sort in alphabetical order for having a stable deterministic order
label_list = sorted(pd.concat([train_df["next_activity"], val_df["next_activity"], test_df["next_activity"]]).unique())
LABEL_INDEX = {lbl: i for i, lbl in enumerate(label_list)} # A map from label string → integer index in that stable order.
labels_for_prompt = "\n".join(label_list) # A single string listing labels, newline-separated, for prompt insertion

# training-set log priors (in fixed label order)
# Compute p(label) on train only to avoid peeking at val/test (data leakage). ensures the prior vector aligns exactly with the stable label_list order (labels seen only in val/test will be NaN here)
# .fillna(1e-8) gives tiny mass to labels that never occurred in TRAIN
train_label_freq = train_df["next_activity"].value_counts(normalize=True).reindex(label_list).fillna(1e-8) 
# Calibration trick that often stabilizes predictions on imbalanced label sets. So labels common in TRAIN get a small additive bump; rare/never-seen labels get less or none. Because you tune prior_alpha on VAL, you can control how strong this bias is.
LOG_PRIOR = np.log(train_label_freq.values.astype(np.float32))

# Build a TRAIN-ONLY bigram map: last_event -> set of observed next labels
# Purpose:
#   Inject minimal process knowledge into zero-shot predictions by
#   restricting candidate labels to those that have actually followed
#   the last event in the training split (no leakage).
# Notes:
#   - This is a hard filter (prunes candidates). If coverage is too
#     aggressive for rare tails, consider softening (e.g., downweight).
#   - If the last event wasn't seen in TRAIN, we fall back to all labels.
NEXTS = defaultdict(set)
for _, r in train_df.iterrows():
    p = r["prefix"]
    if p:
        NEXTS[p[-1]].add(r["next_activity"])

alpha = 0.5
bigram_counts = defaultdict(Counter)
for _, r in train_df.iterrows():
    p = r["prefix"]
    if p:
        bigram_counts[p[-1]][r["next_activity"]] += 1

unigram_ctr = Counter(train_df["next_activity"])
uni_Z = sum(unigram_ctr[l] + alpha for l in label_list)
LOG_P_NEXT = {l: np.log((unigram_ctr[l] + alpha) / uni_Z) for l in label_list}

LOG_P_NEXT_GIVEN_LAST = {}
for last, ctr in bigram_counts.items():
    Z = sum(ctr[l] + alpha for l in label_list)
    LOG_P_NEXT_GIVEN_LAST[last] = {l: np.log((ctr[l] + alpha) / Z) for l in label_list}

tri_counts = defaultdict(Counter)
for _, r in train_df.iterrows():
    p = r["prefix"]
    if len(p) >= 2:
        tri_counts[(p[-2], p[-1])][r["next_activity"]] += 1

def trigram_logprior(last2, cands, alpha=0.5):
    ctr = tri_counts.get(tuple(last2))
    if not ctr:
        return None
    Z = sum(ctr[l] + alpha for l in label_list)
    return np.array([np.log((ctr[l] + alpha) / Z) for l in cands], dtype=np.float32)

# %% Model / Tokenizer
MODEL_NAME = FS_CFG["model_name"]
DTYPE = torch.float16 if (torch.cuda.is_available() and FS_CFG["dtype"]=="fp16") else torch.float32
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and FS_CFG["device"]=="auto") else "cpu")

# Loads the GPT-Neo tokenizer
# Ensures there is a pad_token_id
# Sets left truncation so if anything must be trimmed, you keep the most recent tokens
# Caches the tokenizer’s EOS id.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left" 
eos_id = tokenizer.eos_token_id

# Model & budget
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, low_cpu_mem_usage=True, torch_dtype=DTYPE
).to(DEVICE).eval()

# Retrieves the model’s max sequence length (2048 for GPT-Neo). The guard handles exotic sentinel values some tokenizers expose.
MAX_TOK = getattr(tokenizer, "model_max_length", 2048)
if MAX_TOK is None or MAX_TOK > 10**8:  # guard weird sentinel values
    MAX_TOK = 2048

# Build per-label token id sequences used for exact log-likelihood scoring.
# Leading space is critical for GPT-style byte-BPE tokenizers: it makes label tokenization consistent with a normal completion after a space (prevents unfair token splits).
# Optional EOS after the label helps stop generation cleanly and reduces continuation bias (especially when one label is a prefix of another).
# Pre-materializes label token tensors on the target DEVICE so you don’t repeatedly copy them during scoring.
LABEL_IDS = {
    lbl: tokenizer(" " + lbl, add_special_tokens=False).input_ids
         + ([eos_id] if FS_CFG["add_eos_after_label"] else [])
    for lbl in label_list
}
LABEL_TENSORS = {lbl: torch.tensor(ids, dtype=torch.long, device=DEVICE) for lbl, ids in LABEL_IDS.items()}
# Score [prompt + label] in one forward pass. To avoid over-length sequences, you reserve max_label_len + 8 tokens of headroom.max(256, …) adds a safety floor so the prompt never becomes absurdly tiny even if labels are long.
max_label_len = max(len(ids) for ids in LABEL_IDS.values()) if LABEL_IDS else 0
PROMPT_BUDGET = max(256, MAX_TOK - (max_label_len + 8))  # safety floor

# %% Retrieval for ICL demos (TF-IDF + MMR diversity)
def _seq_str(pfx): return " ".join(pfx)
train_df["prefix_str"] = train_df["prefix"].apply(_seq_str)
tfidf       = TfidfVectorizer().fit(train_df["prefix_str"])
train_tfidf = tfidf.transform(train_df["prefix_str"])

def retrieve_demos(prefix, n_shots, max_demo_events, mmr_lambda=0.5):
    q_last = prefix[-1] if prefix else None
    # Prefer train rows whose last in-prefix equals q_last
    subset = train_df
    if q_last is not None:
        subset = train_df[train_df["prefix"].apply(lambda p: len(p)>0 and p[-1]==q_last)]
        if subset.empty:
            subset = train_df

    tfidf_local = tfidf.transform(subset["prefix"].apply(_seq_str))
    q = tfidf.transform([_seq_str(prefix)])
    sims_q = cosine_similarity(q, tfidf_local).ravel()
    pool_idx = np.argsort(sims_q)[::-1][: n_shots*10]

    chosen, chosen_vecs, seen = [], [], set()
    for _ in range(n_shots*3):  # oversample to achieve label diversity
        best_idx, best_score = None, -1e9
        for ridx in pool_idx:
            if any(ridx == c for _, _, c in chosen): 
                continue
            v = tfidf_local[ridx]
            rel = sims_q[ridx]
            div = 0.0 if not chosen_vecs else max(cosine_similarity(v, vv).ravel()[0] for vv in chosen_vecs)
            mmr = mmr_lambda * rel - (1.0 - mmr_lambda) * div
            g = subset.iloc[ridx]["next_activity"]
            if g in seen: 
                continue
            if mmr > best_score:
                best_score, best_idx = mmr, ridx
        if best_idx is None: break
        ex = subset.iloc[best_idx]
        chosen.append((ex["prefix"][-max_demo_events:], ex["next_activity"], best_idx))
        chosen_vecs.append(tfidf_local[best_idx])
        seen.add(ex["next_activity"])
        if len(chosen) >= n_shots: break

    if not chosen:
        return []
    return [(p, g) for (p, g, _) in chosen][:n_shots]

# %% Candidate pruning: Precompute global label freq (you already have train_label_freq)
label_freq_counter = Counter(train_df["next_activity"])

def candidate_labels(prefix, K):
    last = prefix[-1] if len(prefix) else None
    if FS_CFG["use_bigram_filter"] and last in NEXTS and NEXTS[last]:
        allowed = sorted(NEXTS[last], key=lambda l: (-label_freq_counter[l], l))
        if FS_CFG.get("no_self_loop_if_unseen", False) and last not in NEXTS.get(last, set()):
            allowed = [a for a in allowed if a != last] or allowed
        # relax: if too few, backfill by global freq
        if len(allowed) < K:
            backfill = [a for a, _ in label_freq_counter.most_common(K*2) if a not in allowed]
            return (allowed + backfill)[:K] or [label_list[0]]
        return allowed[:K]
    # fallback: global frequent
    cands = [a for a, _ in label_freq_counter.most_common(K)]
    return cands or [label_list[0]]

# %% Prompt building — ZS-aligned cache (IDs only), safe key (query tail + cands + demos)
def _trace_str(seq): return FS_CFG["event_sep"].join(seq)

def build_fs_prompt(prefix, cands, demos):
    labs_txt = "\n".join(cands)
    blocks = []
    for p_demo, gold in demos:
        blocks.append(
            FS_CFG["prompt_tmpl_demo"].format(
                trace=_trace_str(p_demo),
                labels=labs_txt,
                gold=gold
            )
        )
    q = prefix[-FS_CFG["ctx_events"]:]
    blocks.append(
        FS_CFG["prompt_tmpl_query"].format(
            trace=_trace_str(q),
            labels=labs_txt
        )
    )
    return "".join(blocks)

# cache ONLY tokenized prompt IDs on-device
# _PROMPT_CACHE = {}  # key -> 1D LongTensor on DEVICE
_PROMPT_CACHE = OrderedDict()
_PROMPT_CACHE_CAP = 2048

def _prompt_cache_key(prefix, cands, demos):
    q_tail = tuple(prefix[-FS_CFG["ctx_events"]:])
    cands_key = tuple(cands)
    demos_key = tuple((tuple(p), g) for (p, g) in demos)  # order matters
    return (q_tail, cands_key, demos_key)

def get_prompt_ids(prefix, cands, demos):
    """
    Build chat messages and apply Qwen chat template, then tokenize (no label).
    """
    key = _prompt_cache_key(prefix, cands, demos)
    if key in _PROMPT_CACHE:
        _PROMPT_CACHE.move_to_end(key)
        return _PROMPT_CACHE[key]
    prompt = build_fs_prompt(prefix, cands, demos)
    base_ids = tokenizer(prompt + " ", add_special_tokens=False,
                         truncation=True, max_length=PROMPT_BUDGET).input_ids
    base_ids_tensor = torch.tensor(base_ids, dtype=torch.long, device=DEVICE)
    _PROMPT_CACHE[key] = base_ids_tensor
    if len(_PROMPT_CACHE) > _PROMPT_CACHE_CAP:
        _PROMPT_CACHE.popitem(last=False)
    return base_ids_tensor

def make_cands_and_demos(prefix, K, n_shots):
    # candidates (as you have)
    if FS_CFG["use_bigram_filter"] and len(prefix):
        allowed = list(NEXTS.get(prefix[-1], set()))
        base_labels = allowed if allowed else label_list
        if len(base_labels):
            freq_sorted = sorted(base_labels, key=lambda l: float(train_label_freq.get(l, 0.0)), reverse=True)
            cands = freq_sorted[:K] if len(freq_sorted) > K else freq_sorted
        else:
            cands = candidate_labels(prefix, K=K)
    else:
        cands = candidate_labels(prefix, K=K)
    raw_demos = retrieve_demos(prefix, n_shots, FS_CFG["max_demo_events"])
    demos = [d for d in raw_demos if d[1] in cands] or []
    # if no matching demos, prefer *empty demos* over contradictory ones
    if not demos:
        demos = []  # keep label list self-consistent; no demos is better than mismatched demos
    demos = [(p[-FS_CFG["max_demo_events"]:], g) for (p, g) in demos]
    return cands, demos

# %% Scoring (batch over pruned candidates)
# 1) cands + demos (no tokenization)
# 2) ZS-style cached tokenization
# 3) batch scoring
def _softmax_np(x):
    x = np.array(x, dtype=np.float32); m = x.max(); e = np.exp(x - m); s = e.sum()
    return (e/s) if s > 0 else np.ones_like(x)/len(x)

@torch.no_grad()
def score_cands(prefix, K=None, n_shots=None):
    K = FS_CFG["K_prune"] if K is None else K
    n_shots = FS_CFG["n_shots"] if n_shots is None else n_shots
    cands, demos = make_cands_and_demos(prefix, K, n_shots)
    if not len(cands):
        return np.array([]), np.array([]), []
    base_ids = get_prompt_ids(prefix, cands, demos)
    rows, lens = [], []
    for lbl in cands:
        L = LABEL_TENSORS[lbl]
        rows.append(torch.cat([base_ids, L], dim=0))
        lens.append(int(L.size(0)))
    pad = tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad)
    attn = (input_ids != pad)  # bool mask
    logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
    cut = base_ids.size(0)
    scores = []
    for i, lbl in enumerate(cands):
        L = lens[i]
        lp  = torch.log_softmax(logits[i, cut-1:cut-1+L, :], dim=-1)
        tgt = LABEL_TENSORS[lbl]
        s   = lp.gather(-1, tgt.unsqueeze(-1)).sum()
        if FS_CFG["length_norm"] and L > 0:
            s = s / L
        scores.append(float(s))
    scores = np.array(scores, dtype=np.float32)

    # temperature
    scores = scores / max(1e-6, FS_CFG["temperature"])

    # trigram backoff prior (optional, safe)
    if len(prefix) >= 2:
        tri = trigram_logprior(prefix[-2:], cands)
        if tri is not None:
            scores = scores + 0.8 * tri  # weight to tune: [0.4, 0.6, 0.8, 1.0]

    # global class prior (weak)
    if FS_CFG["use_class_prior"] and FS_CFG["prior_alpha"] > 0:
        prior = np.array([LOG_PRIOR[LABEL_INDEX[l]] for l in cands], dtype=np.float32)
        scores = scores + FS_CFG["prior_alpha"] * prior
    # soft bigram boost
    if FS_CFG.get("bigram_weight", 0.0) > 0 and len(prefix):
        last = prefix[-1]
        cond = LOG_P_NEXT_GIVEN_LAST.get(last)
    if cond is None:
        cond_vec = np.array([LOG_P_NEXT[l] for l in cands], dtype=np.float32)
    else:
        cond_vec = np.array([cond.get(l, LOG_P_NEXT[l]) for l in cands], dtype=np.float32)
    scores = scores + FS_CFG["bigram_weight"] * cond_vec

    probs = _softmax_np(scores)
    return scores, probs, cands

def predict_topk(prefix_tokens, k=5):
    scores, probs, cands = score_cands(prefix_tokens)
    if len(scores) == 0:
        return None, [], 0.0, []
    idx = np.argsort(scores)[-k:][::-1]
    labels_k = [cands[i] for i in idx]
    probs_k  = [float(probs[i]) for i in idx]
    return labels_k[0], labels_k, probs_k[0], probs_k

# %% Tiny validation sweep (optional)
def tune_on_val():
    if not FS_CFG["do_val_tune"]:
        return
    y_true = val_df["next_activity"].tolist()
    best = (-1.0, FS_CFG["temperature"], FS_CFG["prior_alpha"], FS_CFG["ctx_events"],
            FS_CFG["K_prune"], FS_CFG["n_shots"])
    for ctx in FS_CFG.get("ctx_events_grid", [FS_CFG["ctx_events"]]):
      for K in FS_CFG["grid_K"]:
        for nS in FS_CFG["grid_shots"]:
          FS_CFG["ctx_events"], FS_CFG["K_prune"], FS_CFG["n_shots"] = ctx, K, nS
          for t in FS_CFG["grid_taus"]:
            for a in FS_CFG["grid_alphas"]:
              FS_CFG["temperature"], FS_CFG["prior_alpha"] = t, a
              y_pred = [predict_topk(p, 1)[0] for p in val_df["prefix"]]
              acc = accuracy_score(y_true, y_pred)
              if acc > best[0]:
                  best = (acc, t, a, ctx, K, nS)
    FS_CFG["temperature"], FS_CFG["prior_alpha"], FS_CFG["ctx_events"], FS_CFG["K_prune"], FS_CFG["n_shots"] = best[1:6]
    wandb.config.update({"FS_CFG_tuned": {
        "val_acc": best[0], "temperature": FS_CFG["temperature"], "prior_alpha": FS_CFG["prior_alpha"],
        "ctx_events": FS_CFG["ctx_events"], "K_prune": FS_CFG["K_prune"], "n_shots": FS_CFG["n_shots"]
    }}, allow_val_change=True)
    
tune_on_val()

# Record final knobs + a prompt sample (debug)
if len(val_df):
    ex = val_df.sample(1, random_state=config["seed"]).iloc[0]
    cands_s, demos_s = make_cands_and_demos(ex["prefix"], FS_CFG["K_prune"], FS_CFG["n_shots"])
    pstr = build_fs_prompt(ex["prefix"], cands_s, demos_s)
    wandb.config.update({
        "n_labels": len(label_list),
        "labels_for_prompt": "\n".join(label_list),
        "prompt_template_demo": FS_CFG["prompt_tmpl_demo"],
        "prompt_template_query": FS_CFG["prompt_tmpl_query"],
        "final_ctx_events": FS_CFG["ctx_events"],
        "final_temperature": FS_CFG["temperature"],
        "final_prior_alpha": FS_CFG["prior_alpha"]
        }, allow_val_change=True)

# %% Per-k loop over actual k values; compute macro averages over k; micro Accuracy
k_vals, accuracies, fscores, precisions, recalls, counts = [], [], [], [], [], []

for i in sorted(test_df["k"].astype(int).unique()):
    test_data_subset = test_df[test_df["k"] == i]
    if len(test_data_subset) > 0:
        y_true = test_data_subset["next_activity"].tolist()
        prefixes = test_data_subset["prefix"].tolist()  # these are lists of strings
        y_pred = [predict_topk(p, k=1)[0] for p in prefixes]  # get top-1 prediction per prefix
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        k_vals.append(i)
        counts.append(len(y_true))
        accuracies.append(accuracy)
        fscores.append(fscore)
        precisions.append(precision)
        recalls.append(recall)

avg_accuracy = float(np.mean(accuracies)) if accuracies else float("nan")
avg_f1 = float(np.mean(fscores)) if fscores else float("nan")
avg_precision = float(np.mean(precisions)) if precisions else float("nan")
avg_recall = float(np.mean(recalls)) if recalls else float("nan")

print(f"Average accuracy across all prefixes:  {avg_accuracy:.4f}")
print(f"Average f-score across all prefixes:   {avg_f1:.4f}")
print(f"Average precision across all prefixes: {avg_precision:.4f}")
print(f"Average recall across all prefixes:    {avg_recall:.4f}") 

# Micro (global) accuracy over all test prefixes
y_true_val = val_df["next_activity"].tolist()
prefixes_val = val_df["prefix"].tolist()
y_pred_all = [predict_topk(p, k=1)[0] for p in prefixes_val]
micro_acc_val = accuracy_score(y_true_val, y_pred_all)
print(f"[VAL]  Micro (global) accuracy: {micro_acc_val:.4f}")

# Micro (global) accuracy over all test prefixes
y_true_test = test_df["next_activity"].tolist()
prefixes_test = test_df["prefix"].tolist()
y_pred_all = [predict_topk(p, k=1)[0] for p in prefixes_test]
micro_acc = accuracy_score(y_true_test, y_pred_all)
print(f"[TEST] Micro (global) accuracy: {micro_acc:.4f}")

# %% Plots → disk
os.makedirs(config["plots_dir"], exist_ok=True)

# %% Top-k accuracy on the whole test set 
def topk_accuracy(y_true, topk_labels_list, k=3):
    hits = sum(y_true[i] in topk_labels_list[i][:k] for i in range(len(y_true)))
    return hits / len(y_true) if len(y_true) else float("nan")

topk_all = [predict_topk(p, k=5)[1] for p in test_df["prefix"]]
y_all    = test_df["next_activity"].tolist()
wandb.log({
    "metrics/top3_acc": float(topk_accuracy(y_all, topk_all, k=3)),
    "metrics/top5_acc": float(topk_accuracy(y_all, topk_all, k=5)),
})

# %% Acc/F1 vs k
if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, accuracies, marker="o", label="Accuracy")
    plt.title("Accuracy vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("Accuracy")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(config['plots_dir'], f"acc_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, fscores, marker="o", label="F1 (weighted)")
    plt.title("F1 vs. Prefix Length (k)")
    plt.xlabel("Prefix Length (k)"); plt.ylabel("F1 (weighted)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(config['plots_dir'], f"f1_vs_k_{ts}.png"), dpi=150); plt.close()

print(f"Saved plots to: {config['plots_dir']}")

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
})

# %% Robust confusion matrix
def _norm(s): return str(s).strip()

y_true_lbl = [_norm(x) for x in test_df["next_activity"].tolist()]
prefixes_test = test_df["prefix"].tolist()
y_pred_lbl = [_norm(predict_topk(p, k=1)[0]) for p in prefixes_test]
cm_labels = label_list

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
    cm_path = os.path.join(config['plots_dir'], f"confusion_matrix_{ts}.png")
    plt.savefig(cm_path, dpi=150); plt.close()
    wandb.log({"cm_image": wandb.Image(cm_path)})

# %% Samples table
sample = test_df.sample(n=min(5, len(test_df)), random_state=config["seed"]) if len(test_df) else test_df
table = wandb.Table(columns=["k", "prefix", "gold", "pred", "p_pred", "top5", "top5_p"])

for _, r in sample.iterrows():
    toks = r["prefix"] if isinstance(r["prefix"], list) else str(r["prefix"]).split()
    pred, top5, p_pred, top5_p = predict_topk(toks, k=5)
    
    prefix_pretty = " → ".join(toks)
    gold = str(r["next_activity"])
    
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

# %% Finish
wandb.finish()