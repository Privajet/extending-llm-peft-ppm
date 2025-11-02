# %% Zero-shot Next-Activity (ACT) with GPT-Neo-1.3B — tuned prompt + priors
# - Multi-class classification, but done generatively by scoring candidate labels with a causal LM.
# - strict label scoring via log-likelihood of [prompt + label + EOS]
# - short, recent context (last-N events)
# - temperature + optional class-prior calibration
# - optional bigram candidate mask from train set
# - small validation sweep for temperature/prior_alpha
# - W&B logging of metrics, curves, prompt, and full config

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
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

# %% 
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %%
DATASET = "HelpDesk"

config = {
    # bookkeeping
    "dataset":                  DATASET,
    "plots_dir":                f"/ceph/lfertig/Thesis/notebook/{DATASET}/plots/gpt-neo-1.3B/ZS/ACT"
}

ZS_CFG = {
    # model / runtime
    "family":                   "neo",
    "model_name":               "EleutherAI/gpt-neo-1.3B",
    "dtype":                    "fp16",                                 # "fp32" if CPU-only
    "device":                   "auto",
    # prompt & context
    "ctx_events":               12,                                     # last N events from prefix: Grid: [6, 8, 12, 16]
    "event_sep":                " → ",
    "prompt_tmpl":              (
                                "Trace: {trace}\n"
                                "Choose EXACTLY ONE label from the list below and output ONLY that label.\n"
                                "Labels:\n{labels}\n"
                                "Answer:"
                                ),
    "add_eos_after_label":      True,                                   # append EOS after label to stop continuation
    # scoring
    "length_norm":              True,                                   # divide label log-likelihood by label token length
    "temperature":              0.85,                                   # tune on val; grid {0.7, 0.85, 1.0, 1.2}
    "use_class_prior":          True,                                   # add alpha * log p(label) from train
    "prior_alpha":              0.25,                                   # tune on val; grid {0.0, 0.15, 0.25, 0.4}
    # transition knowledge (soft / hard)
    "use_bigram_boost":         True,                                   # soft bump for labels seen after last event in TRAIN
    "bigram_boost":             0.30,
    "use_bigram_filter":        False,                                  # restrict candidates to labels seen after last event in train
    # validation sweep (tiny)
    "do_val_tune":              True,
    "grid_taus":                [0.6, 0.75, 0.9, 1.0, 1.15],
    "grid_alphas":              [0.0, 0.15, 0.25, 0.35, 0.5],
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
    project=f"gpt-neo-1.3B_ACT_ZeroShot_{config['dataset']}",
    entity="privajet-university-of-mannheim",
    name=f"neo_zs_act_{ts}",
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

# Generate stable label list
label_list = sorted(pd.concat([train_df["next_activity"], val_df["next_activity"], test_df["next_activity"]]).unique())
LABEL_INDEX = {lbl: i for i, lbl in enumerate(label_list)}
labels_for_prompt = "\n".join(label_list)

# Training-set log priors (in fixed label order)
# Compute class prior logp(y) for optional calibration
# Ensures the prior vector aligns exactly with the stable label_list order (labels seen only in val/test will be NaN here)
# .fillna(1e-8) gives tiny mass to labels that never occurred in TRAIN
# Calibration trick that often stabilizes predictions on imbalanced label sets. So labels common in TRAIN get a small additive bump; rare/never-seen labels get less or none. Because you tune prior_alpha on VAL, you can control how strong this bias is.
train_label_freq = train_df["next_activity"].value_counts(normalize=True).reindex(label_list).fillna(1e-8) 
LOG_PRIOR = np.log(train_label_freq.values.astype(np.float32))

# Build bigram map: last_event → {feasible next acts}
# Prefix is a list of events; you take only the last N events (recency).
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
    if len(p) == 0: 
        continue
    last = p[-1]
    NEXTS[last].add(r["next_activity"])

# %% Model / Tokenizer
MODEL_NAME = ZS_CFG["model_name"]
DTYPE = torch.float16 if (torch.cuda.is_available() and ZS_CFG["dtype"]=="fp16") else torch.float32
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and ZS_CFG["device"]=="auto") else "cpu")

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
         + ([eos_id] if ZS_CFG["add_eos_after_label"] else [])
    for lbl in label_list
}
LABEL_TENSORS = {lbl: torch.tensor(ids, dtype=torch.long, device=DEVICE) for lbl, ids in LABEL_IDS.items()}

# Score [prompt + label] in one forward pass. To avoid over-length sequences, you reserve max_label_len + 8 tokens of headroom.max(256, …) adds a safety floor so the prompt never becomes absurdly tiny even if labels are long.
max_label_len = max(len(ids) for ids in LABEL_IDS.values()) if LABEL_IDS else 0
PROMPT_BUDGET = max(256, MAX_TOK - (max_label_len + 8))  # safety floor

# %% Prompt builder (ZS-only, no demos)
# cache ONLY tokenized prompt IDs on DEVICE (like your FS refactor)
# _PROMPT_CACHE memoizes tokenized prompt IDs keyed by the exact last-N events (tuple). This avoids repeated tokenization and host→device copies.
# Keep only the last ctx_events to emphasize recency and control length.
# Serialize the trace with a deterministic separator (event_sep).
_PROMPT_CACHE = {} # key -> 1D LongTensor on DEVICE

def build_prompt(prefix_tokens):
    t = prefix_tokens[-ZS_CFG["ctx_events"]:]
    trace = ZS_CFG["event_sep"].join(t)
    # Fill a fixed template:
    return ZS_CFG["prompt_tmpl"].format(trace=trace, labels=labels_for_prompt)

def get_prompt_ids(prefix_tokens):
    key = tuple(prefix_tokens[-ZS_CFG["ctx_events"]:])
    if key in _PROMPT_CACHE:
        return _PROMPT_CACHE[key]
    prompt = build_prompt(prefix_tokens)
    P = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=PROMPT_BUDGET).input_ids
    P_ids = torch.tensor(P, dtype=torch.long, device=DEVICE)
    _PROMPT_CACHE[key] = P_ids
    return P_ids

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
# For each candidate label y in the label list:
#  1. Build text = prompt + “ ” + label (+ EOS)
#  2. Run one forward pass and compute sum of token log-probs for the label substring.
#  3. Length normalization (divide by label token length) to avoid bias for short labels. 
@torch.no_grad()
def score_labels(prefix_tokens):
    P_ids = get_prompt_ids(prefix_tokens)
    # candidate pruning via bigram map (from TRAIN set); fallback to all labels
    if ZS_CFG["use_bigram_filter"] and len(prefix_tokens):
        cand_set = list(NEXTS.get(prefix_tokens[-1], set()))
        cands = [lbl for lbl in label_list if lbl in cand_set] or label_list
    else:
        cands = label_list
    rows, lens, idx_map = [], [], []
    for lbl in cands:
        L = LABEL_TENSORS[lbl]
        rows.append(torch.cat([P_ids, L], dim=0))
        lens.append(len(L))
        idx_map.append(LABEL_INDEX[lbl])
    pad_id = tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad_id)
    attn = (input_ids != pad_id).long()
    logits = model(input_ids=input_ids, attention_mask=attn).logits
    cut = P_ids.size(0)
    ll = []
    for i, Llen in enumerate(lens):
        lp = torch.log_softmax(logits[i, cut-1:cut-1+Llen, :], dim=-1)
        tgt = LABEL_TENSORS[cands[i]]
        s = lp.gather(-1, tgt.unsqueeze(-1)).sum()
        if ZS_CFG["length_norm"] and Llen > 0:
            s = s / Llen
        ll.append(float(s))
    ll = np.array(ll, dtype=np.float32)

    # scatter back into full label array; apply temperature
    full_scores = np.full(len(label_list), -1e9, dtype=np.float32)
    full_scores[idx_map] = ll / max(1e-6, ZS_CFG["temperature"])

    # optional class-prior bias
    if ZS_CFG["use_class_prior"]:
        full_scores += ZS_CFG["prior_alpha"] * LOG_PRIOR

    # optional soft bigram boost
    if ZS_CFG["use_bigram_boost"] and len(prefix_tokens):
        allowed = NEXTS.get(prefix_tokens[-1], None)
        if allowed:
            boost = np.zeros(len(label_list), dtype=np.float32)
            idxs = [LABEL_INDEX[l] for l in allowed if l in LABEL_INDEX]
            if idxs:
                boost[idxs] = ZS_CFG["bigram_boost"]
                full_scores += boost

    # numerically-stable softmax
    # Convert scores → probabilities via stable softmax; pick top-k.
    m = np.max(full_scores)
    probs = np.exp(full_scores - m); s = probs.sum()
    probs = probs / s if s > 0 else probs
    return full_scores, probs

def predict_topk(prefix_tokens, k=5):
    scores, probs = score_labels(prefix_tokens)
    idx = np.argsort(scores)[-k:][::-1]
    labels_k = [label_list[i] for i in idx]
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
    "final_ctx_events": ZS_CFG["ctx_events"],
    "final_temperature": ZS_CFG["temperature"],
    "final_prior_alpha": ZS_CFG["prior_alpha"],
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