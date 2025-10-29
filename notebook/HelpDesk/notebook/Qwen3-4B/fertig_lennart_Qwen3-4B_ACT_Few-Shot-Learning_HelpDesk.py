# %% Few-shot (ICL) Next-Activity (ACT) with Qwen3-4B — chat template + exact label scoring
# - strict label scoring via log-likelihood of [prompt + label (+EOS)]
# - TF-IDF retrieval for few-shot demos (MMR diversified), temporal split (no leakage)
# - candidate pruning via TRAIN bigrams (top-K), optional soft bigram boost / hard filter
# - length normalization, temperature, class-prior calibration (from TRAIN)
# - tiny validation sweep on VAL for tau/prior_alpha/K_prune/n_shots/ctx_events
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter, defaultdict, OrderedDict

import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

# %% Repro + logging
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)
log.info("PyTorch: %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available(): log.info("GPU: %s", torch.cuda.get_device_name(0))

# %% W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# %% Configs (mirrors ZS layout)
RUN_CFG = {
    "seed": SEED,
    "case_col": "case:concept:name",
    "act_col":  "concept:name",
    "time_col": "time:timestamp",
    "plots_dir": "/ceph/lfertig/Thesis/notebook/HelpDesk/plots/Qwen3-4B/FS/ACT",
    "unit": "days",
}

FS_CFG = {
    # model / runtime
    "family": "qwen",
    "model_name": "Qwen/Qwen3-4B-Instruct-2507",
    "dtype": "fp16",
    "device": "auto",
    # prompt & context (few-shot)
    "n_shots": 5,               # grid: [3,5]
    "ctx_events": 12,           # query-tail events
    "max_demo_events": 10,      # grid: [6,8,10]
    "event_sep": " → ",
    # Chat system instruction to force single-label outputs
    "system_msg": (
        "You are an assistant for next-activity prediction. "
        "Given a trace and a label list, output EXACTLY ONE label from the list. "
        "No extra text, punctuation, or explanations."
    ),
    # demo/query block templates (pure text that goes inside a chat turn)
    "prompt_tmpl_demo": (
        "Trace: {trace}\n"
        "Labels:\n{labels}\n"
        "Answer: {gold}\n\n"
    ),
    "prompt_tmpl_query": (
        "Trace: {trace}\n"
        "Labels:\n{labels}\n"
        "Answer:"
    ),
    # scoring & calibration
    "add_eos_after_label": True,
    "length_norm": True,
    "temperature": 0.7,
    "use_class_prior": False,
    "prior_alpha": 0.00,
    "bigram_weight": 1.2,       # soft P(next|last) boost
    # transition knowledge
    "use_bigram_boost": True,
    "bigram_boost": 0.30,
    "use_bigram_filter": True,  # hard prune to labels seen after last
    # pruning
    "K_prune": 8,               # grid: [6,8,12]
    "no_self_loop_if_unseen": True,
    # validation sweep (tiny)
    "do_val_tune": False,
    "grid_taus":   [0.6, 0.7, 0.8],
    "grid_alphas": [0.0, 0.1],
    "grid_K":      [6, 8, 12],
    "grid_shots":  [3, 5],
    "ctx_events_grid": [8, 12, 16],
    # evaluation
    "topk": [1, 3, 5],
}

run = wandb.init(
    project="Qwen3-4B_ACT_FewShot_HelpDesk",
    entity="privajet-university-of-mannheim",
    name=f"Qwen3-4B_icl_act_{ts}",
    config={"run_cfg": RUN_CFG, "fs_cfg": FS_CFG},
)

# %% Data
train_df = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_train.csv")
val_df   = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_val.csv")
test_df  = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_test.csv")

for d in (train_df, val_df, test_df):
    d.rename(columns={"next_act": "next_activity"}, inplace=True)
    d["prefix"] = d["prefix"].astype(str).str.split()

print(f"Train prefixes: {len(train_df)} - Validation prefixes: {len(val_df)} - Test prefixes: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# labels (stable order)
label_list = sorted(pd.concat([train_df["next_activity"], val_df["next_activity"], test_df["next_activity"]]).unique())
LABEL_INDEX = {lbl: i for i, lbl in enumerate(label_list)}
labels_for_prompt = "\n".join(label_list)

# TRAIN-only priors
train_label_freq = train_df["next_activity"].value_counts(normalize=True).reindex(label_list).fillna(1e-8)
LOG_PRIOR = np.log(train_label_freq.values.astype(np.float32))

# TRAIN-only transitions (bigrams + trigrams + backoff)
NEXTS = defaultdict(set)
bigram_counts = defaultdict(Counter)
unigram_ctr = Counter(train_df["next_activity"])
for _, r in train_df.iterrows():
    p = r["prefix"]
    if p:
        NEXTS[p[-1]].add(r["next_activity"])
        bigram_counts[p[-1]][r["next_activity"]] += 1

alpha = 0.5
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
torch_dtype = torch.float16 if (torch.cuda.is_available() and FS_CFG["dtype"]=="fp16") else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    device_map="auto",
    trust_remote_code=True,     # Qwen family often needs this
).eval()
# Loads the Qwen tokenizer
# Ensures there is a pad_token_id
# Sets left truncation so if anything must be trimmed, you keep the most recent tokens
# Caches the tokenizer’s EOS id.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"
eos_id = tokenizer.eos_token_id
DEVICE = next(model.parameters()).device

# Context budget (chat template can be long; keep headroom for label)
MAX_TOK = tokenizer.model_max_length if tokenizer.model_max_length and tokenizer.model_max_length < 10**9 else 8192

# Build per-label token id sequences used for exact log-likelihood scoring.
# - Prepend a leading space: Qwen3-4b use byte-level BPE where tokenization
#   depends on word boundary. After "...Assistant: " the next token begins after a
#   space, so we mirror that here to get consistent, fair tokenization.
# - Optionally append EOS: encourages a clean "label + stop" completion, which
#   helps when some labels are prefixes of others or benefit from continuation bias.
#   Keep length normalization on to compare across differing token counts.
LABEL_IDS = {
    lbl: tokenizer(" " + lbl, add_special_tokens=False).input_ids
         + ([eos_id] if FS_CFG["add_eos_after_label"] else [])
    for lbl in label_list
}
LABEL_TENSORS = {lbl: torch.tensor(ids, dtype=torch.long, device=DEVICE) for lbl, ids in LABEL_IDS.items()}
max_label_len = max(len(ids) for ids in LABEL_IDS.values()) if LABEL_IDS else 0
PROMPT_BUDGET = max(256, MAX_TOK - (max_label_len + 8))  # keep room for label

# %% Prompt builder (chat template)
def _seq_str(pfx): return " ".join(pfx)
train_df["prefix_str"] = train_df["prefix"].apply(_seq_str)
tfidf       = TfidfVectorizer().fit(train_df["prefix_str"])

def retrieve_demos(prefix, n_shots, max_demo_events, mmr_lambda=0.5):
    q_last = prefix[-1] if prefix else None
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
    for _ in range(n_shots*3):
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
        if len(allowed) < K:
            backfill = [a for a, _ in label_freq_counter.most_common(K*2) if a not in allowed]
            return (allowed + backfill)[:K] or [label_list[0]]
        return allowed[:K]
    return [a for a, _ in label_freq_counter.most_common(K)] or [label_list[0]]

# %% Prompt building — ZS-aligned cache (IDs only), safe key (query tail + cands + demos)
def _trace_str(seq): return FS_CFG["event_sep"].join(seq)

def build_fs_text(prefix, cands, demos):
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
    demos_key = tuple((tuple(p), g) for (p, g) in demos)
    return (q_tail, cands_key, demos_key)

def get_prompt_ids(prefix, cands, demos):
    """
    Build chat messages and apply Qwen chat template, then tokenize (no label).
    """
    key = _prompt_cache_key(prefix, cands, demos)
    if key in _PROMPT_CACHE:
        _PROMPT_CACHE.move_to_end(key)
        return _PROMPT_CACHE[key]
    user_text = build_fs_text(prefix, cands, demos)
    messages = [
        {"role": "system", "content": FS_CFG["system_msg"]},
        {"role": "user", "content": user_text}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # will end with Assistant:
    )
    # Budget: trim tokens to PROMPT_BUDGET
    base_ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=PROMPT_BUDGET).input_ids
    base_ids_tensor = torch.tensor(base_ids, dtype=torch.long, device=DEVICE)

    _PROMPT_CACHE[key] = base_ids_tensor
    if len(_PROMPT_CACHE) > _PROMPT_CACHE_CAP:
        _PROMPT_CACHE.popitem(last=False)
    return base_ids_tensor

def make_cands_and_demos(prefix, K, n_shots):
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
    if not demos:
        demos = []
    demos = [(p[-FS_CFG["max_demo_events"]:], g) for (p, g) in demos]
    return cands, demos

# %% Scoring (batch over pruned candidates)
def _softmax_np(x):
    x = np.array(x, dtype=np.float32); m = x.max(); e = np.exp(x - m); s = e.sum()
    return (e/s) if s > 0 else np.ones_like(x)/len(x)

@torch.no_grad()
def score_with_ll(base_ids, cands, label_tensors, tokenizer, model, length_norm=True):
    pad = tokenizer.pad_token_id
    rows, lens = [], []
    for lbl in cands:
        L = label_tensors[lbl]
        rows.append(torch.cat([base_ids, L], dim=0))
        lens.append(int(L.size(0)))
    input_ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad).to(base_ids.device)
    attn = (input_ids != pad).to(base_ids.device)
    
    logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
    cut = base_ids.size(0)

    scores = []
    for i, lbl in enumerate(cands):
        L = lens[i]
        lp  = torch.log_softmax(logits[i, cut-1:cut-1+L, :], dim=-1)
        tgt = label_tensors[lbl]
        s   = lp.gather(-1, tgt.unsqueeze(-1)).sum()
        if length_norm and L > 0:
            s = s / L
        scores.append(float(s))
    return np.array(scores, dtype=np.float32)

def fs_score(prefix, K=None, n_shots=None):
    K = FS_CFG["K_prune"] if K is None else K
    n_shots = FS_CFG["n_shots"] if n_shots is None else n_shots
    cands, demos = make_cands_and_demos(prefix, K, n_shots)
    if not cands: return np.array([]), np.array([]), []

    base_ids = get_prompt_ids(prefix, cands, demos)  # FS-specific (depends on cands+demos)
    scores = score_with_ll(base_ids, cands, LABEL_TENSORS, tokenizer, model, FS_CFG["length_norm"])

    # identical post-processing in both ZS/FS:
    scores = scores / max(1e-6, FS_CFG["temperature"])
    if len(prefix) >= 2:
        tri = trigram_logprior(prefix[-2:], cands)
        if tri is not None:
            scores += 0.8 * tri
    if FS_CFG["use_class_prior"] and FS_CFG["prior_alpha"] > 0:
        scores += FS_CFG["prior_alpha"] * np.array([LOG_PRIOR[LABEL_INDEX[l]] for l in cands], np.float32)
    if FS_CFG.get("bigram_weight", 0.0) > 0 and len(prefix):
        last = prefix[-1]
        cond = LOG_P_NEXT_GIVEN_LAST.get(last)
        cond_vec = np.array([ (cond.get(l, LOG_P_NEXT[l]) if cond else LOG_P_NEXT[l]) for l in cands], np.float32)
        scores += FS_CFG["bigram_weight"] * cond_vec

    probs = np.exp(scores - scores.max()); probs /= probs.sum() if probs.sum()>0 else 1
    return scores, probs, cands

def predict_topk(prefix_tokens, k=5):
    scores, probs, cands = fs_score(prefix_tokens)
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
    log.info("Tuned on val → acc=%.4f, tau=%.2f, alpha=%.2f", best[0], best[1], best[2])

tune_on_val()

# Record final knobs + a prompt sample (debug)
if len(val_df):
    ex = val_df.sample(1, random_state=SEED).iloc[0]
    cands_s, demos_s = make_cands_and_demos(ex["prefix"], FS_CFG["K_prune"], FS_CFG["n_shots"])
    pstr = build_fs_text(ex["prefix"], cands_s, demos_s)
    wandb.config.update({
        "n_labels": len(label_list),
        "labels_for_prompt": "\n".join(label_list),
        "prompt_template_demo": FS_CFG["prompt_tmpl_demo"],
        "prompt_template_query": FS_CFG["prompt_tmpl_query"],
        "final_ctx_events": FS_CFG["ctx_events"],
        "final_temperature": FS_CFG["temperature"],
        "final_prior_alpha": FS_CFG["prior_alpha"],
        "system_msg": FS_CFG["system_msg"],
        "prompt_sample": pstr[:1000]
        }, allow_val_change=True)

# %% Per-k evaluation
k_vals, accuracies, fscores, precisions, recalls, counts = [], [], [], [], [], []

for k in sorted(test_df["k"].astype(int).unique()):
    test_data_subset = test_df[test_df["k"] == k]
    if len(test_data_subset) > 0:
        y_true = test_data_subset["next_activity"].tolist()
        y_pred = [predict_topk(p, k=1)[0] for p in test_data_subset["prefix"]]
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        k_vals.append(k); counts.append(len(test_data_subset))
        accuracies.append(acc); precisions.append(prec); recalls.append(rec); fscores.append(f1)

avg_accuracy = float(np.mean(accuracies)) if accuracies else float("nan")
avg_f1  = float(np.mean(fscores))    if fscores    else float("nan")
avg_precision   = float(np.mean(precisions)) if precisions else float("nan")
avg_recall   = float(np.mean(recalls))    if recalls    else float("nan")

print(f"Average accuracy across all prefixes:  {avg_accuracy:.4f}")
print(f"Average f-score across all prefixes:   {avg_f1:.4f}")
print(f"Average precision across all prefixes: {avg_precision:.4f}")
print(f"Average recall across all prefixes:    {avg_recall:.4f}")

# Micro (global) accuracy over all val prefixes
# Total correct / total samples across the entire val set
y_true_all = val_df["next_activity"].tolist()
y_pred_all = [predict_topk(p, k=1)[0] for p in val_df["prefix"]]
micro_acc  = accuracy_score(y_true_all, y_pred_all)
print(f"[Val] Micro (global) accuracy: {micro_acc:.4f}")

# Micro (global) accuracy over all test prefixes
# Total correct / total samples across the entire test set
y_true_all = test_df["next_activity"].tolist()
y_pred_all = [predict_topk(p, k=1)[0] for p in test_df["prefix"]]
micro_acc  = accuracy_score(y_true_all, y_pred_all)
print(f"[TEST] Micro (global) accuracy: {micro_acc:.4f}")

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

# %% Plots
plot_dir = RUN_CFG["plots_dir"]
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

# %% Samples table
sep = FS_CFG.get("event_sep", " → ")
sample = test_df.sample(n=min(5, len(test_df)), random_state=SEED) if len(test_df) else test_df
table = wandb.Table(columns=["k","prefix","gold","pred","p_pred","top3","top3_p"])
for _, r in sample.iterrows():
    pred, topk, p_pred, topk_p = predict_topk(r["prefix"], k=3)
    prefix_str = sep.join(r["prefix"])
    gold = r["next_activity"]
    print("Prefix:", prefix_str)
    print("Gold:  ", gold)
    print(f"Pred:  {pred} ({p_pred:.3f})")
    print("Top-3:", topk)
    print("-"*60)
    table.add_data(
        int(r["k"]),
        prefix_str,
        gold,
        pred,
        float(p_pred),
        ", ".join(topk),
        ", ".join(f"{x:.3f}" for x in topk_p),
    )
wandb.log({"samples": table})

# %% Finish
wandb.finish()