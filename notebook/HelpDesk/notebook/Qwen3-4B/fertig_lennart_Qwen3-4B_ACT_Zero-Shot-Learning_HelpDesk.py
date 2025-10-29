# %% Zero-shot Next-Activity (ACT) with Qwen3-4B — chat template + exact label scoring
# - strict label scoring via log-likelihood of [chat_prompt + label (+ EOS)]
# - short, recent context (last-N events), left truncation (Left truncation keeps the most recent events when the prompt must be shortened.)
# - temperature + optional class-prior calibration
# - optional bigram candidate mask from train set (soft or hard)
# - tiny validation sweep for tau/prior_alpha
# - W&B logging of metrics, curves, prompt, and config
# - How: Convert the problem into label ranking. For every possible next activity, compute how likely the model is to continue the prompt with that exact label, then pick the best.

import os, sys, glob, ctypes, random, logging
os.environ["MPLBACKEND"] = "Agg"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# try to preload libstdc++ (useful on some HPC stacks)
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
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter, defaultdict

# %% Reproducibility & logging
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

# %%  W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# base run config
RUN_CFG = {
    "seed": SEED,
    "case_col": "case:concept:name",
    "act_col":  "concept:name",
    "time_col": "time:timestamp",
    "plots_dir": "/ceph/lfertig/Thesis/notebook/HelpDesk/plots/Qwen3-4B/ZS/ACT",
    "unit": "days",
}

# zero-shot “hyperparameters” (prompt & scoring)
ZS_CFG = {
    # model / runtime
    "family": "qwen",
    "model_name": "Qwen/Qwen3-4B-Instruct-2507",
    "dtype": "fp16",
    "device": "auto",
    # prompt & context (few-shot)
    "ctx_events": 12,           # query-tail events
    "event_sep": " → ",
    # Chat system instruction to force single-label outputs
    "system_msg": (
        "You are an assistant for next-activity prediction. "
        "Given a trace and a label list, output EXACTLY ONE label from the list. "
        "No extra text, punctuation, or explanations."
    ),
    # demo/query block templates (pure text that goes inside a chat turn)
    "prompt_tmpl": (
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
    "use_bigram_filter": True,  # hard prune to labels seen after last
    # pruning
    "K_prune": 8,               # grid: [6,8,12]
    "no_self_loop_if_unseen": True,
    # validation sweep (tiny)
    "do_val_tune": True,
    "grid_taus":   [0.6, 0.7, 0.8],
    "grid_alphas": [0.0, 0.1],
    "ctx_events_grid": [8, 12, 16],
    # evaluation
    "topk": [1, 3, 5],
}

run = wandb.init(
    project="Qwen3-4B_ACT_ZeroShot_HelpDesk",
    entity="privajet-university-of-mannheim",
    name=f"Qwen3-4B_zeroshot_act_{ts}",
    config={**RUN_CFG, "zs_cfg": ZS_CFG},
)

# %% Data
train_df = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_train.csv")
val_df   = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_val.csv")
test_df  = pd.read_csv("/ceph/lfertig/Thesis/data/HelpDesk/processed/next_activity_test.csv")

for d in (train_df, val_df, test_df):
    d.rename(columns={"next_act": "next_activity"}, inplace=True)
    d["prefix"] = d["prefix"].astype(str).str.split() # convert space-separated strings to lists

print(f"Train prefixes: {len(train_df)} - Validation prefixes: {len(val_df)} - Test prefixes: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# labels: stable alphabetical order (use union of splits)
label_list = sorted(pd.concat([train_df["next_activity"], val_df["next_activity"], test_df["next_activity"]]).unique())
LABEL_INDEX = {lbl: i for i, lbl in enumerate(label_list)}
labels_for_prompt = "\n".join(label_list)

# training-set log priors (in fixed label order)
train_label_freq = train_df["next_activity"].value_counts(normalize=True).reindex(label_list).fillna(1e-8)
LOG_PRIOR = np.log(train_label_freq.values.astype(np.float32))

# Build a TRAIN-ONLY bigram map: NEXTS[last_event] = {labels ever seen after that event in TRAIN}
# Purpose:
#   Inject minimal process knowledge into zero-shot predictions by
#   restricting candidate labels to those that have actually followed
#   the last event in the training split (no leakage).
# Notes:
#   - This is a hard filter (prunes candidates). If coverage is too
#     aggressive for rare tails, consider softening (e.g., downweight).
#   - If the last event wasn't seen in TRAIN, we fall back to all labels.
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
MODEL_NAME = ZS_CFG["model_name"]

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
    tokenizer.pad_token = tokenizer.eos_token  # safe padding
tokenizer.truncation_side = "left"
eos_id = tokenizer.eos_token_id

# Model & budget
# parity with Neo: fp16 on GPU, fp32 otherwise; place with device_map="auto"
torch_dtype = torch.float16 if (torch.cuda.is_available() and ZS_CFG["dtype"]=="fp16") else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    device_map="auto",
    trust_remote_code=True,     # Qwen family often needs this
).eval()

DEVICE = next(model.parameters()).device

# Context budget (chat template can be long; keep headroom for label)
MAX_TOK = getattr(tokenizer, "model_max_length", 2048)
if MAX_TOK is None or MAX_TOK > 10**8:  # guard weird sentinel values
    MAX_TOK = 2048

# Build per-label token id sequences used for exact log-likelihood scoring.
# - Prepend a leading space: Qwen3-4b use byte-level BPE where tokenization
#   depends on word boundary. After "...Assistant: " the next token begins after a
#   space, so we mirror that here to get consistent, fair tokenization.
# - Optionally append EOS: encourages a clean "label + stop" completion, which
#   helps when some labels are prefixes of others or benefit from continuation bias.
#   Keep length normalization on to compare across differing token counts.
LABEL_IDS = {
    lbl: tokenizer(" " + lbl, add_special_tokens=False).input_ids
         + ([eos_id] if (eos_id is not None and ZS_CFG["add_eos_after_label"]) else [])
    for lbl in label_list
}
LABEL_TENSORS = {
    lbl: torch.tensor(ids, dtype=torch.long, device=DEVICE)
    for lbl, ids in LABEL_IDS.items()
}
max_label_len = max(len(x) for x in LABEL_IDS.values()) if LABEL_IDS else 0
PROMPT_BUDGET = max(256, MAX_TOK - (max_label_len + 8))

# %% Prompt builder (chat template)
def _user_content_from_tmpl(tmpl, trace_str, labels_str):
    return tmpl.format(trace=trace_str, labels=labels_str)

# Wrap that user content with Qwen’s chat template (tokenizer.apply_chat_template(..., add_generation_prompt=True)), which is important for instruction-tuned models.
def build_chat_prompt(prefix_tokens):
    t = prefix_tokens[-ZS_CFG["ctx_events"]:]
    trace = ZS_CFG["event_sep"].join(t)
    content = _user_content_from_tmpl(ZS_CFG["prompt_tmpl"], trace, labels_for_prompt)
    messages = [
        {"role": "system", "content": ZS_CFG["system_msg"]},
        {"role": "user", "content": content}
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return chat_text

_PROMPT_CACHE = {}
def get_prompt_ids(prefix_tokens):
    key = tuple(prefix_tokens[-ZS_CFG["ctx_events"]:])
    if key in _PROMPT_CACHE:
        return _PROMPT_CACHE[key]
    chat_text = build_chat_prompt(prefix_tokens)
    P = tokenizer(chat_text, add_special_tokens=False, truncation=True, max_length=PROMPT_BUDGET).input_ids
    P_ids = torch.tensor(P, dtype=torch.long, device=DEVICE)
    _PROMPT_CACHE[key] = P_ids
    return P_ids

# %% Candidate pruning: Precompute global label freq (you already have train_label_freq)
label_freq_counter = Counter(train_df["next_activity"])

def candidate_labels(prefix, K):
    last = prefix[-1] if len(prefix) else None
    if ZS_CFG["use_bigram_filter"] and last in NEXTS and NEXTS[last]:
        allowed = sorted(NEXTS[last], key=lambda l: (-label_freq_counter[l], l))
        if ZS_CFG.get("no_self_loop_if_unseen", False) and last not in NEXTS.get(last, set()):
            allowed = [a for a in allowed if a != last] or allowed
        if len(allowed) < K:
            backfill = [a for a, _ in label_freq_counter.most_common(K*2) if a not in allowed]
            return (allowed + backfill)[:K] or [label_list[0]]
        return allowed[:K]
    return [a for a, _ in label_freq_counter.most_common(K)] or [label_list[0]]

# log the *exact* prompt template and current labels
wandb.config.update({
    "final_ctx_events": ZS_CFG["ctx_events"],
    "final_temperature": ZS_CFG["temperature"],
    "final_prior_alpha": ZS_CFG["prior_alpha"],
    "n_labels": len(label_list),
    "prompt_template": ZS_CFG["prompt_tmpl"],
    "labels_for_prompt": labels_for_prompt
}, allow_val_change=True)

# %% Scoring
@torch.no_grad()
def score_with_ll(base_ids, cands, label_tensors, tokenizer, model, length_norm=True):
    pad = tokenizer.pad_token_id
    rows, lens = [], []
    for lbl in cands:
        L = label_tensors[lbl]
        rows.append(torch.cat([base_ids, L], dim=0))
        lens.append(int(L.size(0)))
    input_ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad)
    attn = (input_ids != pad)
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

def zs_score(prefix):
    # choose candidate labels using the same pruning policy as FS (top-K with optional bigram filter)
    K = ZS_CFG.get("K_prune", None)
    if K is not None:
        cands = candidate_labels(prefix, K=K)   # uses NEXTS + global freq + no_self_loop_if_unseen if requested
    else:
        if ZS_CFG["use_bigram_filter"] and len(prefix):
            cand_set = list(NEXTS.get(prefix[-1], set()))
            cands = [l for l in label_list if l in cand_set] or label_list
        else:
            cands = label_list

    base_ids = get_prompt_ids(prefix)  # ZS-specific: prompt depends only on the query
    scores = score_with_ll(base_ids, cands, LABEL_TENSORS, tokenizer, model, ZS_CFG["length_norm"])

    # identical post-processing as FS
    scores = scores / max(1e-6, ZS_CFG["temperature"])

    if len(prefix) >= 2:
        tri = trigram_logprior(prefix[-2:], cands)
        if tri is not None:
            scores += 0.8 * tri

    if ZS_CFG.get("use_class_prior", False) and ZS_CFG.get("prior_alpha", 0) > 0:
        scores += ZS_CFG["prior_alpha"] * np.array([LOG_PRIOR[LABEL_INDEX[l]] for l in cands], np.float32)

    if ZS_CFG.get("bigram_weight", 0.0) > 0 and len(prefix):
        last = prefix[-1]
        cond = LOG_P_NEXT_GIVEN_LAST.get(last)
        cond_vec = np.array([(cond.get(l, LOG_P_NEXT[l]) if cond else LOG_P_NEXT[l]) for l in cands], np.float32)
        scores += ZS_CFG["bigram_weight"] * cond_vec

    probs = np.exp(scores - scores.max())
    Z = probs.sum()
    probs = probs / Z if Z > 0 else probs
    return scores, probs, cands

def predict_topk(prefix_tokens, k=5):
    sc, probs, cands = zs_score(prefix_tokens)
    idx = np.argsort(sc)[-k:][::-1]
    labels_k = [cands[i] for i in idx]
    probs_k  = [float(probs[i]) for i in idx]
    return labels_k[0], labels_k, probs_k[0], probs_k

# %% Tiny validation sweep (optional)
def tune_on_val():
    if not ZS_CFG["do_val_tune"]:
        return
    y_true = val_df["next_activity"].tolist()
    best = (-1.0, ZS_CFG["temperature"], ZS_CFG["prior_alpha"], ZS_CFG["ctx_events"])

    for ctx in ZS_CFG.get("ctx_events_grid", [ZS_CFG["ctx_events"]]):
        ZS_CFG["ctx_events"] = ctx
        for t in ZS_CFG["grid_taus"]:
            for a in ZS_CFG["grid_alphas"]:
                ZS_CFG["temperature"] = t
                ZS_CFG["prior_alpha"] = a
                y_pred = [predict_topk(p, 1)[0] for p in val_df["prefix"]]
                acc = accuracy_score(y_true, y_pred)
                if acc > best[0]:
                    best = (acc, t, a, ctx)

    ZS_CFG["temperature"], ZS_CFG["prior_alpha"], ZS_CFG["ctx_events"] = best[1], best[2], best[3]
    wandb.config.update(
        {"zs_cfg_tuned": {"val_acc": best[0], "temperature": best[1], "prior_alpha": best[2], "ctx_events": best[3]}},
        allow_val_change=True
    )
    log.info("Tuned on val → acc=%.4f, tau=%.2f, alpha=%.2f", best[0], best[1], best[2])

tune_on_val()
wandb.config.update({
    "system_msg": ZS_CFG["system_msg"],
    "final_ctx_events": ZS_CFG["ctx_events"],
    "final_temperature": ZS_CFG["temperature"],
    "final_prior_alpha": ZS_CFG["prior_alpha"],
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

# %% Robust confusion matrix (avoid KeyError by normalizing strings + union classes)
def _norm(s): 
    return str(s).strip()

# Gather true/pred labels as strings
y_true_lbl = [_norm(x) for x in test_df["next_activity"].tolist()]
y_pred_lbl = [_norm(predict_topk(p, k=1)[0]) for p in test_df["prefix"]]

# Consistent label axis (union of observed classes), or use label_list for a fixed axis
cm_labels = sorted(set(y_true_lbl) | set(y_pred_lbl))
# If you prefer a stable axis across runs, do:
# cm_labels = label_list

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
    
# %% Samples table
sample = test_df.sample(n=min(5, len(test_df)), random_state=SEED) if len(test_df) else test_df
table = wandb.Table(columns=["k","prefix","gold","pred","p_pred","top3","top3_p"])
for _, r in sample.iterrows():
    pred, topk, p_pred, topk_p = predict_topk(r["prefix"], k=3)
    print("Prefix:", " → ".join(r["prefix"]))
    print("Gold:  ", r["next_activity"])
    print(f"Pred:  {pred} ({p_pred:.3f})")
    print("Top-3:", topk)
    print("-"*60)
    table.add_data(
        r["k"],
        " → ".join(r["prefix"]),
        r["next_activity"],
        pred,
        float(p_pred),
        ", ".join(topk),
        ", ".join([f"{x:.3f}" for x in topk_p])
    )
wandb.log({"samples": table})

# %% Finish
wandb.finish()