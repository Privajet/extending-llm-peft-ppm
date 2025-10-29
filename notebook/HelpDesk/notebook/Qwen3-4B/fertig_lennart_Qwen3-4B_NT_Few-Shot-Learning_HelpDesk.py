# %% Few-shot (ICL) Next-Time (NT) with Qwen3-4B — chat template + exact label scoring
# - Task: predict time-to-next-event by classifying into TRAIN-derived quantile bins (in days)
# - Demos: TF-IDF retrieval with MMR diversification from TRAIN only (temporal split ⇒ no leakage)
# - Prompting: Qwen chat template (system + user), last-N events context (left truncation), strict “ONE bin only”
# - Scoring: exact log-likelihood of [prompt + bin_label (+EOS)] with length normalization
# - Calibration: temperature + global bin prior from TRAIN (+ optional conditional prior p(bin|last_act))
# - Point estimate: probability-weighted expected value over bin midpoints (BIN_MIDS) → predicted days
# - Binning: robust quantile edge construction on TRAIN with min-bin-count and low/high clipping guards
# - Tuning: tiny VAL sweep for {temperature, prior_alpha, cond_alpha, ctx_events} (mirrors ZS knobs)
# - Evaluation: per-k MAE/MSE/RMSE + (optional) top-1 bin accuracy; W&B logging of curves, bins, prompt, config
# - Performance: cache tokenized prompt IDs (with demos) and prebuild label tensors on device to reduce overhead
# - Repro: fixed seed, deterministic CUDA flags; unit is days to match ZS NT pipeline

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
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from collections import OrderedDict

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
    "plots_dir": "/ceph/lfertig/Thesis/notebook/HelpDesk/plots/Qwen3-4B/FS/NT",
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
    "ctx_events": 20,           # query-tail events
    "max_demo_events": 10,      # grid: [6,8,10]
    "event_sep": " → ",
    # Chat system instruction to force single-label outputs
    "system_msg": (
        "You are an assistant for next-time prediction. "
        "Given a trace and a list of time bins (in days), choose EXACTLY ONE bin from the list and output ONLY that bin. "
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
    "temperature": 0.9,
    "use_class_prior": True,
    "prior_alpha": 0.25,
    # optional conditional prior by last activity (train-only, no leakage)
    "use_cond_prior_by_last_act": True,
    "cond_alpha": 0.35,   # strength; 0.2–0.6 often works; tune on val
    # # pruning
    # "K_prune": 8,               # grid: [6,8,12]
    # "no_self_loop_if_unseen": True,
    # validation grids
    "do_val_tune": False,
    "grid_taus":   [0.7, 0.85, 1.0, 1.2],
    "grid_alphas": [0.0, 0.15, 0.25, 0.4],
    "grid_cond":   [0.0, 0.25, 0.35, 0.5],
    "ctx_events_grid": [8, 12, 16],
    # binning
    "num_bins": 20,
    "min_bin_count": 5,
    "clip_low_high": (1e-6, None),
    # "grid_K":      [6, 8, 12],
    # "grid_shots":  [3, 5],
    # evaluation
    # "topk": [1, 3, 5],
}

run = wandb.init(
    project="Qwen3-4B_NT_FewShot_HelpDesk",
    entity="privajet-university-of-mannheim",
    name=f"Qwen3-4B_icl_nt_{ts}",
    config={"run_cfg": RUN_CFG, "fs_cfg": FS_CFG},
)

# %% Data
train_df = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_time_train.csv")
val_df   = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_time_val.csv")
test_df  = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_time_test.csv")

def to_days(series): return series.astype(float)

for d in (train_df, val_df, test_df):
    d["prefix"] = d["prefix"].astype(str).str.split()
    if "next_time_delta" not in d.columns:
        d.rename(columns={"next_time": "next_time_delta"}, inplace=True)
    if "nt_days" not in d.columns:
        d["nt_days"] = to_days(d["next_time_delta"])
    d["last_act"] = d["prefix"].apply(lambda p: p[-1] if len(p) else "[START]")

print(f"Train prefixes: {len(train_df)} - Validation prefixes: {len(val_df)} - Test prefixes: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# %% Quantile bins on TRAIN (NT in days) → labels
def make_bins(train_series, num_bins=20, min_bin_count=5, clip_low=1e-6, clip_high=None):
    def _quantile(x, qs):
        try:    return np.quantile(x, qs, method="nearest")
        except TypeError: return np.quantile(x, qs, interpolation="nearest")

    x = train_series.values.astype(np.float64)
    if clip_low  is not None: x = np.maximum(x, clip_low)
    if clip_high is not None: x = np.minimum(x, clip_high)

    qs = np.linspace(0, 1, num_bins+1)
    edges = np.unique(_quantile(x, qs))
    if len(edges) < 3:
        lo, hi = x.min(), x.max()
        if hi <= lo: hi = lo + 1e-6
        edges = np.exp(np.linspace(np.log(lo), np.log(hi), num_bins+1))

    def enforce_min_count(edges, x_train, min_bin_count):
        if min_bin_count is None or min_bin_count <= 1: return edges
        edges = edges.copy()
        while len(edges) > 2:
            bin_idx = np.digitize(x_train, edges, right=True) - 1
            counts = np.bincount(np.clip(bin_idx, 0, len(edges)-2), minlength=len(edges)-1)
            if counts.min() >= min_bin_count: break
            i = int(np.argmin(counts))
            if i == 0: edges = np.delete(edges, i+1)
            elif i == len(counts)-1: edges = np.delete(edges, i)
            else: edges = np.delete(edges, i+1 if counts[i+1] <= counts[i-1] else i)
        return edges

    edges = enforce_min_count(edges, x, min_bin_count)
    mids, labels = [], []
    for i in range(len(edges)-1):
        L, R = float(edges[i]), float(edges[i+1])
        mids.append((L + R) / 2.0)
        labels.append(f"({L:.5f}, {R:.5f}] days")
    return np.array(edges, dtype=np.float64), np.array(mids, dtype=np.float64), labels

BIN_EDGES, BIN_MIDS, BIN_LABELS = make_bins(
    train_df["nt_days"],
    num_bins=FS_CFG["num_bins"],
    min_bin_count=FS_CFG["min_bin_count"],
    clip_low=FS_CFG["clip_low_high"][0],
    clip_high=FS_CFG["clip_low_high"][1]
)
n_bins = len(BIN_LABELS)
BIN_INDEX = {lbl: i for i, lbl in enumerate(BIN_LABELS)}
COND_DEFAULT = np.log(np.ones(n_bins, dtype=np.float32) / n_bins)

def digitize_nt(x):
    idx = np.digitize(x, BIN_EDGES, right=True) - 1
    return int(np.clip(idx, 0, n_bins-1))

for frame in (train_df, val_df, test_df):
    frame["bin_idx"] = frame["nt_days"].apply(digitize_nt)
    frame["bin_label"] = frame["bin_idx"].apply(lambda i: BIN_LABELS[i])

# Priors
train_bin_freq = train_df["bin_idx"].value_counts(normalize=True).reindex(range(n_bins)).fillna(1e-8)
LOG_PRIOR = np.log(train_bin_freq.values.astype(np.float32))

COND_LOG_PRIOR = {}
if FS_CFG["use_cond_prior_by_last_act"]:
    for act, series in train_df.groupby("last_act")["bin_idx"]:
        freq = series.value_counts(normalize=True).reindex(range(n_bins)).fillna(1e-8).values.astype(np.float32)
        COND_LOG_PRIOR[act] = np.log(freq)

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
    lbl: tokenizer(" " + lbl, add_special_tokens=False).input_ids + ([eos_id] if FS_CFG["add_eos_after_label"] else [])
    for lbl in BIN_LABELS
}
LABEL_TENSORS = {lbl: torch.tensor(ids, dtype=torch.long, device=DEVICE) for lbl, ids in LABEL_IDS.items()}

max_label_len = max(len(ids) for ids in LABEL_IDS.values()) if LABEL_IDS else 0
PROMPT_BUDGET = max(256, MAX_TOK - (max_label_len + 8)) 

# %% Few-shot retrieval (TF-IDF + MMR)
def _seq_str(pfx): return " ".join(pfx)
train_df["prefix_str"] = train_df["prefix"].apply(_seq_str)
tfidf = TfidfVectorizer().fit(train_df["prefix_str"])
train_tfidf = tfidf.transform(train_df["prefix_str"])

def retrieve_demos(prefix, n_shots, max_demo_events, mmr_lambda=0.5):
    q = tfidf.transform([_seq_str(prefix)])
    sims_q = cosine_similarity(q, train_tfidf).ravel()
    short_idx = np.argsort(sims_q)[-n_shots*10:][::-1]

    chosen, chosen_vecs = [], []
    for _ in range(n_shots):
        best_idx, best_mmr = None, -1e9
        for idx in short_idx:
            if any(idx == c for _, _, c in chosen): continue
            v = train_tfidf[idx]; rel = sims_q[idx]
            div = 0.0 if not chosen_vecs else max(cosine_similarity(v, vv).ravel()[0] for vv in chosen_vecs)
            mmr = mmr_lambda * rel - (1.0 - mmr_lambda) * div
            if mmr > best_mmr: best_mmr, best_idx = mmr, idx
        if best_idx is None: break
        ex = train_df.iloc[best_idx]
        chosen.append((ex["prefix"][-max_demo_events:], ex["bin_label"], best_idx))
        chosen_vecs.append(train_tfidf[best_idx])

    if not chosen:
        topk = np.argsort(sims_q)[-n_shots*2:][::-1]
        out, seen = [], set()
        for ridx in topk:
            ex = train_df.iloc[ridx]; g = ex["bin_label"]
            if g in seen: continue
            out.append((ex["prefix"][-max_demo_events:], g))
            seen.add(g)
            if len(out) >= n_shots: break
        return out

    return [(p, g) for p, g, _ in chosen]

# %% FS prompt building + cache
labels_for_prompt = "\n".join(BIN_LABELS)

def _trace_str(seq): 
    return FS_CFG["event_sep"].join(seq)

def build_fs_text(prefix, demos):
    # demos: list of (prefix_list, gold_bin_label)
    labs_txt = labels_for_prompt
    blocks = []
    for p_demo, gold in demos:
        blocks.append(
            FS_CFG["prompt_tmpl_demo"].format(
                trace=_trace_str(p_demo[-FS_CFG["max_demo_events"]:]),
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

def build_chat_prompt(prefix_tokens, demos):
    user_text = build_fs_text(prefix_tokens, demos)
    messages = [
        {"role": "system", "content": FS_CFG["system_msg"]},
        {"role": "user",   "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

_PROMPT_CACHE = OrderedDict()
_PROMPT_CACHE_CAP = 2048

def _prompt_cache_key(prefix, demos):
    # include ctx_events to protect against tuning changes
    ctx = FS_CFG["ctx_events"]
    q_tail = tuple(prefix[-ctx:])
    demos_key = tuple((tuple(p), g) for (p, g) in demos)
    return (ctx, q_tail, demos_key)

def get_prompt_ids(prefix_tokens, demos):
    key = _prompt_cache_key(prefix_tokens, demos)
    if key in _PROMPT_CACHE:
        _PROMPT_CACHE.move_to_end(key)
        return _PROMPT_CACHE[key]
    chat_text = build_chat_prompt(prefix_tokens, demos)
    enc = tokenizer(chat_text, add_special_tokens=False, truncation=True, max_length=PROMPT_BUDGET).input_ids
    P_ids = torch.tensor(enc, dtype=torch.long, device=DEVICE)
    _PROMPT_CACHE[key] = P_ids
    if len(_PROMPT_CACHE) > _PROMPT_CACHE_CAP:
        _PROMPT_CACHE.popitem(last=False)
    return P_ids

# Log exact templates & bins
wandb.config.update({
    "system_msg": FS_CFG["system_msg"],
    "prompt_template_demo": FS_CFG["prompt_tmpl_demo"],
    "prompt_template_query": FS_CFG["prompt_tmpl_query"],
    "n_bins": n_bins,
    "bin_edges": [float(x) for x in BIN_EDGES],
    "bin_labels": BIN_LABELS
}, allow_val_change=True)

# %% Scoring (batch over pruned candidates)
@torch.no_grad()
def score_bins(prefix_tokens, last_act=None):
    demos = retrieve_demos(prefix_tokens, FS_CFG["n_shots"], FS_CFG["max_demo_events"])
    P_ids = get_prompt_ids(prefix_tokens, demos)

    rows, lens = [], []
    for lbl in BIN_LABELS:
        L = LABEL_TENSORS[lbl]
        rows.append(torch.cat([P_ids, L], dim=0))
        lens.append(int(L.size(0)))

    pad_id = tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad_id).to(DEVICE)
    attn = (input_ids != pad_id).long()

    logits = model(input_ids=input_ids, attention_mask=attn).logits.float()

    cut = P_ids.size(0)
    ll = []
    for i, Llen in enumerate(lens):
        lp = torch.log_softmax(logits[i, cut-1:cut-1+Llen, :], dim=-1)
        tgt = LABEL_TENSORS[BIN_LABELS[i]]
        s = lp.gather(-1, tgt.unsqueeze(-1)).sum()
        if FS_CFG["length_norm"] and Llen > 0:
            s = s / Llen
        ll.append(float(s))
    scores = np.array(ll, dtype=np.float32)

    # temperature + priors
    scores = scores / max(1e-6, FS_CFG["temperature"])
    if FS_CFG["use_class_prior"]:
        scores += FS_CFG["prior_alpha"] * LOG_PRIOR
    if FS_CFG["use_cond_prior_by_last_act"]:
        scores += FS_CFG["cond_alpha"] * COND_LOG_PRIOR.get(last_act, COND_DEFAULT)

    # softmax → probs
    m = np.max(scores); probs = np.exp(scores - m); s = probs.sum()
    probs = probs / s if s > 0 else probs
    return scores, probs

def predict_time(prefix_tokens, last_act=None, return_topk=3):
    scores, probs = score_bins(prefix_tokens, last_act)
    if scores.size == 0:
        return float("nan"), [], []
    idx_sorted = np.argsort(scores)[::-1]
    pred_days = float(np.sum(probs * BIN_MIDS))  # probability-weighted midpoint
    k = min(return_topk, len(BIN_LABELS))
    top_idx = idx_sorted[:k]
    top_bins  = [BIN_LABELS[i] for i in top_idx]
    top_probs = [float(probs[i]) for i in top_idx]
    return pred_days, top_bins, top_probs

# %% Tiny validation sweep (tau/prior/cond/ctx)
def tune_on_val():
    if not FS_CFG["do_val_tune"] or len(val_df) == 0:
        return
    y_true = val_df["nt_days"].values.astype(np.float64)

    best = (np.inf, FS_CFG["temperature"], FS_CFG["prior_alpha"], FS_CFG["cond_alpha"], FS_CFG["ctx_events"])
    for ctx in FS_CFG.get("ctx_events_grid", [FS_CFG["ctx_events"]]):
        FS_CFG["ctx_events"] = ctx
        for t in FS_CFG["grid_taus"]:
            for a in FS_CFG["grid_alphas"]:
                for c in FS_CFG["grid_cond"]:
                    FS_CFG["temperature"] = t
                    FS_CFG["prior_alpha"] = a
                    FS_CFG["cond_alpha"] = c
                    preds = []
                    for _, r in val_df.iterrows():
                        p_days, _, _ = predict_time(r["prefix"], r["last_act"])
                        preds.append(p_days)
                    mae = mean_absolute_error(y_true, np.array(preds, dtype=np.float64))
                    if mae < best[0]:
                        best = (mae, t, a, c, ctx)

    FS_CFG["temperature"], FS_CFG["prior_alpha"], FS_CFG["cond_alpha"], FS_CFG["ctx_events"] = best[1], best[2], best[3], best[4]
    wandb.config.update(
        {"FS_CFG_tuned": {"val_mae_days": float(best[0]), "temperature": best[1], "prior_alpha": best[2], "cond_alpha": best[3], "ctx_events": best[4]}},
        allow_val_change=True
    )
    log.info("Tuned on val → MAE=%.4f d, tau=%.2f, alpha=%.2f, cond=%.2f, ctx=%d", best[0], best[1], best[2], best[3], best[4])

tune_on_val()

# record an example prompt
if len(val_df):
    ex = val_df.sample(1, random_state=SEED).iloc[0]
    demos_ex = retrieve_demos(ex["prefix"], FS_CFG["n_shots"], FS_CFG["max_demo_events"])
    prompt_sample = build_chat_prompt(ex["prefix"], demos_ex)[:1200]
    wandb.config.update({"prompt_sample": prompt_sample}, allow_val_change=True)

wandb.config.update({
    "final_ctx_events": FS_CFG["ctx_events"],
    "final_temperature": FS_CFG["temperature"],
    "final_prior_alpha": FS_CFG["prior_alpha"],
    "final_cond_alpha": FS_CFG.get("cond_alpha", 0.0),
    "bin_mids_days": [float(x) for x in BIN_MIDS],
    "unit": RUN_CFG["unit"],
}, allow_val_change=True)

# %% Per-k evaluation
k_vals, counts, maes, mses, rmses = [], [], [], [], []

for k in sorted(test_df["k"].astype(int).unique()):
    subset = test_df[test_df["k"] == k]
    if subset.empty: continue
    y_true = subset["nt_days"].values.astype(np.float64)
    preds = []
    for _, r in subset.iterrows():
        p_days, _, _ = predict_time(r["prefix"], r["last_act"])
        preds.append(p_days)
    preds = np.array(preds, dtype=np.float64)

    k_vals.append(k); counts.append(len(subset))
    mae = mean_absolute_error(y_true, preds)
    mse = mean_squared_error(y_true, preds)
    rmse = float(np.sqrt(mse))
    maes.append(float(mae)); mses.append(float(mse)); rmses.append(rmse)

avg_mae  = float(np.mean(maes))  if maes  else float("nan")
avg_mse  = float(np.mean(mses))  if mses  else float("nan")
avg_rmse = float(np.mean(rmses)) if rmses else float("nan")

print(f"Average MAE across all prefixes:  {avg_mae:.2f} days")
print(f"Average MSE across all prefixes:  {avg_mse:.2f} (days^2)")
print(f"Average RMSE across all prefixes: {avg_rmse:.2f} days")

# (Optional) top-1 bin accuracy (coarse)
y_true_bins = test_df["bin_idx"].values if len(test_df) else np.array([])
y_pred_bins = []
for _, r in test_df.iterrows():
    scores, _ = score_bins(r["prefix"], r["last_act"])
    y_pred_bins.append(int(np.argmax(scores)))
if len(y_pred_bins):
    top1_bin_acc = accuracy_score(y_true_bins, y_pred_bins)
    print(f"Top-1 bin accuracy: {top1_bin_acc:.4f}")
    wandb.log({"metrics/top1_bin_acc": float(top1_bin_acc)})

# %% Plots → disk
plot_dir = RUN_CFG["plots_dir"]
os.makedirs(plot_dir, exist_ok=True)

if len(k_vals):
    plt.figure(figsize=(8,5))
    plt.plot(k_vals, maes, marker='o', label='MAE (days)')
    plt.title('MAE vs. Prefix Length (k)')
    plt.xlabel('Prefix Length (k)'); plt.ylabel('MAE (days)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mae_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, rmses, marker='o', label='RMSE (days)')
    plt.title('RMSE vs. Prefix Length (k)')
    plt.xlabel('Prefix Length (k)'); plt.ylabel('RMSE (days)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"rmse_vs_k_{ts}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, mses, marker='o', label='MSE (days^2)')
    plt.title('MSE vs. Prefix Length (k)')
    plt.xlabel('Prefix Length (k)'); plt.ylabel('MSE (days^2)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mse_vs_k_{ts}.png"), dpi=150); plt.close()

print(f"Saved plots to: {plot_dir}")

# %% Log curves + macro metrics to W&B
wandb.log({
    "curves/k": k_vals,
    "curves/counts": counts,
    "curves/mae": maes,
    "curves/mse": mses,
    "curves/rmse": rmses,
    "metrics/avg_mae":  avg_mae,
    "metrics/avg_mse":  avg_mse,
    "metrics/avg_rmse": avg_rmse,
})

# %% Samples table
sample = test_df.sample(n=min(5, len(test_df)), random_state=SEED) if len(test_df) else test_df
tab = wandb.Table(columns=["k","prefix","last_act","gold_days","pred_days","top_bins","top_probs"])
for _, r in sample.iterrows():
    pred_days, top_bins, top_probs = predict_time(r["prefix"], r["last_act"], return_topk=3)
    print("Trace:", " → ".join(r["prefix"]))
    print(f"Last act: {r['last_act']}")
    print(f"Gold (days): {r['nt_days']:.4f}")
    print(f"Pred (days): {pred_days:.4f}")
    print("Top bins:", top_bins)
    print("-"*60)
    tab.add_data(
        r["k"],
        " → ".join(r["prefix"]),
        r["last_act"],
        float(r["nt_days"]),
        float(pred_days),
        ", ".join(top_bins),
        ", ".join([f"{p:.3f}" for p in top_probs])
    )
wandb.log({"samples": tab})

# %%
wandb.finish()