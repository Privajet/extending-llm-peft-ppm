# %% Zero-shot Next-Activity (ACT) with Llama-3.1-8B — chat template + exact label scoring
# - strict label scoring via log-likelihood of [chat_prompt + label (+ EOT/EOS)]
# - short, recent context (last-N events), left truncation
# - temperature + optional class-prior calibration
# - optional bigram candidate mask from train set (soft or hard)
# - tiny validation sweep for tau/prior_alpha
# - W&B logging of metrics, curves, prompt, and config

import os, sys, glob, ctypes, random, logging
os.environ["MPLBACKEND"] = "Agg"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# preload libstdc++ on some HPC stacks (no-op if not found)
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
    "dataset_path": "/ceph/lfertig/Thesis/data/processed/df_helpdesk.csv.gz",
    "case_col": "case:concept:name",
    "act_col":  "concept:name",
    "time_col": "time:timestamp",
    "plots_dir": "/ceph/lfertig/Thesis/notebook/HelpDesk/plots/Llama-3.1-8B/ZS/ACT",
}

# zero-shot “hyperparameters” (prompt & scoring)
ZS_CFG = {
    # model / runtime
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    # prompt & context
    "ctx_events": 12,           # last N events from prefix: Grid: [6, 8, 12, 16]
    "event_sep": " → ",
    # We build a single user message and wrap it with Llama 3.1 chat template
    "prompt_tmpl": (
        "Trace: {trace}\n"
        "Choose the next activity from the list below.\n"
        "Labels:\n{labels}\n"
        "Answer:"
    ),
    # label termination: for Llama Instruct, <|eot_id|> is typically eos
    "add_eos_after_label": True,
    "prefer_eot_after_label": True,  # use <|eot_id|> if available
    # scoring
    "length_norm": True,        # divide label log-likelihood by label token length
    "temperature": 0.85,        # tune on val; grid {0.7, 0.85, 1.0, 1.2}
    "use_class_prior": True,    # add alpha * log p(label) from train
    "prior_alpha": 0.25,        # tune on val; grid {0.0, 0.15, 0.25, 0.4}
    # transition knowledge (soft/hard)
    "use_bigram_boost": True,   # soft bump for labels seen after last event in TRAIN
    "bigram_boost": 0.30,
    "use_bigram_filter": False, # hard filter candidates to TRAIN next-steps
    # validation sweep (tiny)
    "do_val_tune": True,
    "grid_taus":   [0.6, 0.75, 0.9, 1.0, 1.15],
    "grid_alphas": [0.0, 0.15, 0.25, 0.35, 0.5],
    "ctx_events_grid": [8, 12, 16],
    # evaluation
    "topk": [1, 3, 5],
}

run = wandb.init(
    project="Llama-3.1-8B_ACT_ZeroShot_HelpDesk",
    entity="privajet-university-of-mannheim",
    name=f"Llama31_8B_zeroshot_act_{ts}",
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
max_k = int(pd.concat([train_df["k"], val_df["k"], test_df["k"]]).max()) if len(train_df) + len(val_df) + len(test_df) else 0

# training-set log priors (in fixed label order)
train_label_freq = train_df["next_activity"].value_counts(normalize=True).reindex(label_list).fillna(1e-8)
LOG_PRIOR = np.log(train_label_freq.values.astype(np.float32))

# TRAIN-ONLY bigram map: last_event -> set(next labels)
NEXTS = defaultdict(set)
for _, r in train_df.iterrows():
    p = r["prefix"]
    if len(p) == 0: 
        continue
    last = p[-1]
    NEXTS[last].add(r["next_activity"])

# %% Model / Tokenizer
MODEL_NAME = ZS_CFG["model_name"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token  # safe padding
tokenizer.truncation_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
).eval()

# figure out EOS/EOT for label stopping
eos_id = tokenizer.eos_token_id
eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in tokenizer.get_vocab() else None
label_stop_id = None
if ZS_CFG["add_eos_after_label"]:
    if ZS_CFG["prefer_eot_after_label"] and (eot_id is not None):
        label_stop_id = eot_id
    else:
        label_stop_id = eos_id

# budget: leave headroom for label
MAX_TOK = tokenizer.model_max_length
if MAX_TOK is None or MAX_TOK > 10**8:
    MAX_TOK = 8192
# Build per-label token ids (leading space helps byte-BPE segmentation fairness)
def _label_ids(lbl):
    ids = tokenizer(" " + lbl, add_special_tokens=False).input_ids
    if label_stop_id is not None:
        ids = ids + [label_stop_id]
    return ids

LABEL_IDS = {lbl: _label_ids(lbl) for lbl in label_list}
LABEL_TENSORS = {lbl: torch.tensor(ids, dtype=torch.long, device=model.device) for lbl, ids in LABEL_IDS.items()}
max_label_len = max(len(ids) for ids in LABEL_IDS.values()) if LABEL_IDS else 0
PROMPT_BUDGET = max(256, MAX_TOK - (max_label_len + 8))

# %% Prompt builder (Llama 3.1 chat template)
def _user_text(trace_str, labels_str):
    return ZS_CFG["prompt_tmpl"].format(trace=trace_str, labels=labels_str)

def build_chat_prompt(prefix_tokens):
    # keep recent tail
    t = prefix_tokens[-ZS_CFG["ctx_events"]:]
    trace = ZS_CFG["event_sep"].join(t)
    content = _user_text(trace, labels_for_prompt)
    messages = [
        # you can add a system role here if you want stricter formatting/rules
        {"role": "user", "content": content}
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True  # opens an assistant turn
    )
    return chat_text

# cache only tokenized prompt IDs on device
_PROMPT_CACHE = {}  # key -> 1D LongTensor on device

def get_prompt_ids(prefix_tokens):
    key = tuple(prefix_tokens[-ZS_CFG["ctx_events"]:])
    if key in _PROMPT_CACHE:
        return _PROMPT_CACHE[key]
    chat_text = build_chat_prompt(prefix_tokens)
    P = tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=PROMPT_BUDGET)
    P_ids = P["input_ids"][0].to(model.device)
    _PROMPT_CACHE[key] = P_ids
    return P_ids

# log prompt config to W&B
wandb.config.update({
    "final_ctx_events": ZS_CFG["ctx_events"],
    "final_temperature": ZS_CFG["temperature"],
    "final_prior_alpha": ZS_CFG["prior_alpha"],
    "n_labels": len(label_list),
    "prompt_template_user": ZS_CFG["prompt_tmpl"],
    "labels_for_prompt": labels_for_prompt
}, allow_val_change=True)

# %% Scoring
@torch.no_grad()
def score_labels(prefix_tokens):
    P_ids = get_prompt_ids(prefix_tokens)

    # candidate set (optional bigram filter)
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

    if not rows:
        return np.array([]), np.array([])

    pad_id = tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad_id)
    attn = (input_ids != pad_id).long()

    logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
    cut = P_ids.size(0)

    ll = []
    for i, Llen in enumerate(lens):
        # token-level log-probs for the label span
        lp = torch.log_softmax(logits[i, cut-1:cut-1+Llen, :], dim=-1)
        tgt = LABEL_TENSORS[cands[i]]
        s = lp.gather(-1, tgt.unsqueeze(-1)).sum()
        if ZS_CFG["length_norm"] and Llen > 0:
            s = s / Llen
        ll.append(float(s))
    ll = np.array(ll, dtype=np.float32)

    # scatter back into full label order + temperature
    full_scores = np.full(len(label_list), -1e9, dtype=np.float32)
    full_scores[idx_map] = ll / max(1e-6, ZS_CFG["temperature"])

    # global prior
    if ZS_CFG["use_class_prior"]:
        full_scores += ZS_CFG["prior_alpha"] * LOG_PRIOR

    # soft bigram boost
    if ZS_CFG["use_bigram_boost"] and len(prefix_tokens):
        allowed = NEXTS.get(prefix_tokens[-1], None)
        if allowed:
            boost = np.zeros(len(label_list), dtype=np.float32)
            idxs = [LABEL_INDEX[l] for l in allowed if l in LABEL_INDEX]
            if idxs:
                boost[idxs] = ZS_CFG["bigram_boost"]
                full_scores += boost

    # stable softmax
    m = np.max(full_scores)
    probs = np.exp(full_scores - m)
    s = probs.sum()
    probs = probs / s if s > 0 else probs
    return full_scores, probs

def predict_topk(prefix_tokens, k=5):
    scores, probs = score_labels(prefix_tokens)
    if scores.size == 0:
        return None, [], 0.0, []
    idx = np.argsort(scores)[-k:][::-1]
    labels_k = [label_list[i] for i in idx]
    probs_k  = [float(probs[i]) for i in idx]
    return labels_k[0], labels_k, probs_k[0], probs_k

# %% Tiny validation sweep (optional)
def tune_on_val():
    if not ZS_CFG["do_val_tune"] or len(val_df) == 0:
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
    log.info("Tuned on val → acc=%.4f, tau=%.2f, alpha=%.2f, ctx=%d", best[0], best[1], best[2], best[3])

tune_on_val()
wandb.config.update({
    "final_ctx_events": ZS_CFG["ctx_events"],
    "final_temperature": ZS_CFG["temperature"],
    "final_prior_alpha": ZS_CFG["prior_alpha"],
}, allow_val_change=True)

# %% Per-k evaluation
k_vals, accuracies, fscores, precisions, recalls, counts = [], [], [], [], [], []

for k in sorted(test_df["k"].astype(int).unique()):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue
    y_true = subset["next_activity"].tolist()
    y_pred = [predict_topk(p, 1)[0] for p in subset["prefix"]]
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
})

# %% Top-k accuracy on the whole test set 
def topk_accuracy(y_true, topk_labels_list, k=3):
    hits = sum(y_true[i] in topk_labels_list[i][:k] for i in range(len(y_true)))
    return hits / len(y_true) if len(y_true) else float("nan")

topk_all = [predict_topk(p, k=max(ZS_CFG["topk"]))[1] for p in test_df["prefix"]]
y_all    = test_df["next_activity"].tolist()
wandb.log({
    "metrics/top1_acc": float(topk_accuracy(y_all, topk_all, k=1)),
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

# %% Samples table
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

# %% Finish
wandb.finish()