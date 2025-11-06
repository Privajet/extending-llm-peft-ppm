# %% Zero-shot Remaining-Time (RT) with Qwen3-4B — chat template + quantile bins + priors
# - turn regression (remaining time until case completion) into classification over time bins
# - strict label scoring via log-likelihood of [chat_prompt + time-bin-label (+EOS)]
# - length normalization, temperature, and priors (global + optional conditional by last activity)
# - short, recent context (last-N events), left truncation with prompt budget
# - per-k MAE/RMSE and top-1 bin accuracy, W&B logging (prompt + config)

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

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

# %% W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

# %% 
DATASET = "HelpDesk"

config = {
    # bookkeeping
    "dataset":                  DATASET,
    "plots_dir":                f"/ceph/lfertig/Thesis/notebook/{DATASET}/plots/Qwen3-4B/ZS/RT",
    "unit":                     "days"
}

ZS_CFG = {
    # model / runtime
    "family":                   "qwen",
    "model_name":               "Qwen/Qwen3-4B-Instruct-2507",
    # prompt & context
    "ctx_events":               12,                                     # last N events for context
    "event_sep":                " → ",
    # Chat system instruction to force single-label outputs
    "system_msg":               (
                                "You are an assistant for remaining-time prediction until case completion."
                                "Given a trace and a list of time bins (in days), choose EXACTLY ONE bin from the list and output ONLY that bin."
                                ),
    # demo/query block templates (pure text that goes inside a chat turn)
    "prompt_tmpl":              (
                                "Trace: {trace}\n"
                                "{maybe_elapsed}"
                                "Labels:\n{labels}\n"
                                "Answer:\n\n"
                                ),
    "add_eos_after_label": True,
    # time binning (RT-specific)
    "num_bins":                 20,                                     # number of quantile bins (e.g., 10/20/30)
    "min_bin_count":            5,                                      # safety for degenerate splits
    "clip_low_high":            [1e-6, None],                           # guard tiny zeros to avoid 0-width bins
    # scoring & calibration
    "length_norm":              True,
    "temperature":              0.85,                                   # tuned on val
    "use_class_prior":          True,                                   # global bin prior from TRAIN
    "prior_alpha":              0.25,                                   # strength of global prior
    # optional conditional prior by last activity (train-only, no leakage)
    "use_cond_prior_by_last_act": True,
    "cond_alpha":               0.35,                                   # strength; 0.2–0.6 often works; tune on val
    # tiny validation sweep over a few zero-shot knobs
    "do_val_tune":              False,
    "grid_taus":                [0.7, 0.85, 1.0, 1.2],
    "grid_alphas":              [0.0, 0.15, 0.25, 0.4],
    "grid_cond":                [0.0, 0.25, 0.35, 0.5],                 # cond_alpha candidates
    "ctx_events_grid":          [8, 12, 16]
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
    project=f"Qwen3-4B_RT_ZeroShot_{config['dataset']}",
    entity="privajet-university-of-mannheim",
    name=f"qwen3-4b_zs_rt_{ts}",
    config={**config, "zs_cfg": ZS_CFG},
)

# %% Data
train_df = pd.read_csv(f"/ceph/lfertig/Thesis/data/{config['dataset']}/processed/remaining_time_train.csv")
val_df   = pd.read_csv(f"/ceph/lfertig/Thesis/data/{config['dataset']}/processed/remaining_time_val.csv")
test_df  = pd.read_csv(f"/ceph/lfertig/Thesis/data/{config['dataset']}/processed/remaining_time_test.csv")

def to_days(series):
    return series.astype(float)

for d in (train_df, val_df, test_df):
    d["prefix"] = d["prefix"].astype(str).str.split()

    # Target harmonization -> rt_days
    if "rt_days" not in d.columns:
        if "remaining_time" in d.columns:
            d["rt_days"] = d["remaining_time"].astype(float)
        elif "remaining_time_days" in d.columns:
            d["rt_days"] = d["remaining_time_days"].astype(float)
        elif "remaining_time_delta" in d.columns:
            d["rt_days"] = d["remaining_time_delta"].astype(float)
        else:
            raise ValueError("Cannot find remaining-time target (expected one of: rt_days, remaining_time, "
                             "remaining_time_days, remaining_time_delta)")

    # Optional elapsed time so far -> elapsed_days (used only for the prompt text)
    if "elapsed_days" not in d.columns:
        if "time_passed" in d.columns:
            d["elapsed_days"] = d["time_passed"].astype(float)
        elif "elapsed_time" in d.columns:
            d["elapsed_days"] = d["elapsed_time"].astype(float)
        else:
            # Not available; keep NaN—prompt builder will omit the line
            d["elapsed_days"] = np.nan

    d["last_act"] = d["prefix"].apply(lambda p: p[-1] if len(p) else "[START]")

print(f"Train prefixes: {len(train_df)} - Validation prefixes: {len(val_df)} - Test prefixes: {len(test_df)}")
wandb.log({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

# %% Build quantile bins on TRAIN (RT in days) → labels
def make_bins(train_series, num_bins=20, min_bin_count=5, clip_low=1e-6, clip_high=None):
    def _quantile(x, qs):
        try:    return np.quantile(x, qs, method="nearest")
        except TypeError:  # numpy<1.22
            return np.quantile(x, qs, interpolation="nearest")

    x = train_series.values.astype(np.float64)
    if clip_low  is not None: x = np.maximum(x, clip_low)
    if clip_high is not None: x = np.minimum(x, clip_high)

    qs = np.linspace(0, 1, num_bins+1)
    edges = np.unique(_quantile(x, qs))
    if len(edges) < 3:  # fallback if quantiles collapse
        lo, hi = x.min(), x.max()
        if hi <= lo: hi = lo + 1e-6
        edges = np.exp(np.linspace(np.log(lo), np.log(hi), num_bins+1))

    def enforce_min_count(edges, x_train, min_bin_count):
        if min_bin_count is None or min_bin_count <= 1:
            return edges
        edges = edges.copy()
        while len(edges) > 2:
            bin_idx = np.digitize(x_train, edges, right=True) - 1
            counts = np.bincount(np.clip(bin_idx, 0, len(edges)-2), minlength=len(edges)-1)
            if counts.min() >= min_bin_count:
                break
            i = int(np.argmin(counts))
            # merge towards the smaller neighbor
            if i == 0:
                edges = np.delete(edges, i+1)
            elif i == len(counts)-1:
                edges = np.delete(edges, i)
            else:
                if counts[i+1] <= counts[i-1]:
                    edges = np.delete(edges, i+1)
                else:
                    edges = np.delete(edges, i)
        return edges

    edges = enforce_min_count(edges, x, min_bin_count)

    mids, labels = [], []
    for i in range(len(edges)-1):
        L, R = float(edges[i]), float(edges[i+1])
        mids.append((L + R) / 2.0)
        labels.append(f"({L:.5f}, {R:.5f}] days")
    return np.array(edges, dtype=np.float64), np.array(mids, dtype=np.float64), labels

BIN_EDGES, BIN_MIDS, BIN_LABELS = make_bins(
    train_df["rt_days"],
    num_bins=ZS_CFG["num_bins"],
    min_bin_count=ZS_CFG["min_bin_count"],
    clip_low=ZS_CFG["clip_low_high"][0],
    clip_high=ZS_CFG["clip_low_high"][1]
)
n_bins = len(BIN_LABELS)
BIN_INDEX = {lbl: i for i, lbl in enumerate(BIN_LABELS)}
COND_DEFAULT = np.log(np.ones(n_bins, dtype=np.float32) / n_bins)

def digitize_rt(x):
    idx = np.digitize(x, BIN_EDGES, right=True) - 1  # (L,R]
    return int(np.clip(idx, 0, n_bins-1))

for frame in (train_df, val_df, test_df):
    frame["bin_idx"] = frame["rt_days"].apply(digitize_rt)
    frame["bin_label"] = frame["bin_idx"].apply(lambda i: BIN_LABELS[i])

# Priors (global): train frequency of bins
train_bin_freq = train_df["bin_idx"].value_counts(normalize=True).reindex(range(n_bins)).fillna(1e-8)
LOG_PRIOR = np.log(train_bin_freq.values.astype(np.float32))

# Optional conditional prior by last activity: p(bin | last_act) estimated on TRAIN
COND_LOG_PRIOR = {}
if ZS_CFG["use_cond_prior_by_last_act"]:
    grp = train_df.groupby("last_act")["bin_idx"]
    for act, series in grp:
        freq = series.value_counts(normalize=True).reindex(range(n_bins)).fillna(1e-8).values.astype(np.float32)
        COND_LOG_PRIOR[act] = np.log(freq)

# %% Model / Tokenizer
MODEL_NAME = ZS_CFG["model_name"]

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token  # safe padding
tokenizer.truncation_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True
).eval()

DEVICE = next(model.parameters()).device

eos_id = tokenizer.eos_token_id

# Context budget (chat template can be long; keep headroom for label)
MAX_TOK = tokenizer.model_max_length if tokenizer.model_max_length and tokenizer.model_max_length < 10**9 else 8192

# Label tokenization: add a leading space; optionally append EOS
LABEL_IDS = {
    lbl: tokenizer(" " + lbl, add_special_tokens=False).input_ids
         + ([eos_id] if ZS_CFG["add_eos_after_label"] else [])
    for lbl in BIN_LABELS
}
LABEL_TENSORS = {lbl: torch.tensor(ids, dtype=torch.long, device=DEVICE)
                 for lbl, ids in LABEL_IDS.items()
}
max_label_len = max(len(ids) for ids in LABEL_IDS.values()) if LABEL_IDS else 0
PROMPT_BUDGET = max(256, MAX_TOK - (max_label_len + 8)) 

# %% Prompt builder (chat template)
labels_for_prompt = "\n".join(BIN_LABELS)

def _user_content_from_tmpl(tmpl, trace_str, labels_str, elapsed_days=None):
    show_elapsed = (elapsed_days is not None) and not (isinstance(elapsed_days, float) and np.isnan(elapsed_days))
    maybe_elapsed = f"Elapsed so far: {float(elapsed_days):.3f} days\n" if show_elapsed else ""
    return tmpl.format(trace=trace_str, labels=labels_str, maybe_elapsed=maybe_elapsed)

def build_chat_prompt(prefix_tokens, elapsed_days=None):
    t = prefix_tokens[-ZS_CFG["ctx_events"]:]
    trace = ZS_CFG["event_sep"].join(t)
    content = _user_content_from_tmpl(ZS_CFG["prompt_tmpl"], trace, labels_for_prompt, elapsed_days)
    messages = [
        {"role": "system", "content": ZS_CFG["system_msg"]},
        {"role": "user",   "content": content}
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return chat_text

# cache ONLY tokenized prompt IDs on DEVICE (like your FS refactor)
# _PROMPT_CACHE memoizes tokenized prompt IDs keyed by the exact last-N events (tuple). This avoids repeated tokenization and host→device copies.
_PROMPT_CACHE = {}  # key -> 1D LongTensor on DEVICE

def get_prompt_ids(prefix_tokens, elapsed_days=None):
    tail = tuple(prefix_tokens[-ZS_CFG["ctx_events"]:])
    ed_key = None if (elapsed_days is None or (isinstance(elapsed_days, float) and np.isnan(elapsed_days))) \
             else round(float(elapsed_days), 3)
    key = (tail, ed_key)
    if key in _PROMPT_CACHE:
        return _PROMPT_CACHE[key]
    prompt = build_chat_prompt(prefix_tokens, elapsed_days)
    P = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=PROMPT_BUDGET).input_ids
    P_ids = torch.tensor(P, dtype=torch.long, device=DEVICE)
    _PROMPT_CACHE[key] = P_ids
    return P_ids

# Log exact prompt & bins
wandb.config.update({
    "prompt_template": ZS_CFG["prompt_tmpl"],
    "n_bins": n_bins,
    "bin_edges": [float(x) for x in BIN_EDGES],
    "bin_labels": BIN_LABELS
}, allow_val_change=True)

# %% Scoring (batch over candidate bins)
@torch.no_grad()
def score_bins(prefix_tokens, last_act=None, elapsed_days=None):
    P_ids = get_prompt_ids(prefix_tokens, elapsed_days).unsqueeze(0)
    pad_id = tokenizer.pad_token_id
    attn = (P_ids != pad_id)

    out = model(input_ids=P_ids, attention_mask=attn, use_cache=True)
    base_past   = out.past_key_values
    base_logits = out.logits[:, -1, :]  # logits after the prompt

    ll = []
    for lbl in BIN_LABELS:
        tgt = LABEL_TENSORS[lbl].unsqueeze(0)  # [1, L]
        cur_past   = base_past
        last_logits = base_logits
        s = 0.0
        for t in range(tgt.size(1)):
            # score the current token from last_logits
            s += torch.log_softmax(last_logits, dim=-1).gather(-1, tgt[:, t:t+1]).sum()
            # then feed it to advance cache
            o = model(input_ids=tgt[:, t:t+1], past_key_values=cur_past, use_cache=True)
            cur_past    = o.past_key_values
            last_logits = o.logits[:, -1, :]

        if ZS_CFG["length_norm"] and tgt.size(1) > 0:
            s = s / tgt.size(1)
        ll.append(float(s))

    scores = np.array(ll, dtype=np.float32) / max(1e-6, ZS_CFG["temperature"])
    if ZS_CFG["use_class_prior"]:
        scores += ZS_CFG["prior_alpha"] * LOG_PRIOR
    if ZS_CFG["use_cond_prior_by_last_act"]:
        scores += ZS_CFG["cond_alpha"] * COND_LOG_PRIOR.get(last_act, COND_DEFAULT)

    m = np.max(scores)
    probs = np.exp(scores - m); Z = probs.sum()
    probs = probs / Z if Z > 0 else probs
    return scores, probs

def predict_time(prefix_tokens, last_act=None, elapsed_days=None, return_topk=3):
    scores, probs = score_bins(prefix_tokens, last_act, elapsed_days)
    if scores.size == 0:
        return float("nan"), [], []
    idx_sorted = np.argsort(scores)[::-1]
    # point estimate from probability-weighted midpoints
    pred_days = float(np.sum(probs * BIN_MIDS))
    # top-k bins for analysis
    k = min(return_topk, n_bins)
    top_idx = idx_sorted[:k]
    top_bins = [BIN_LABELS[i] for i in top_idx]
    top_probs = [float(probs[i]) for i in top_idx]
    return pred_days, top_bins, top_probs

# %% Tiny validation sweep (optional) to tune tau/prior/cond
def tune_on_val():
    if not ZS_CFG["do_val_tune"] or len(val_df) == 0:
        return
    y_true = val_df["rt_days"].values.astype(np.float64)

    best = (np.inf, ZS_CFG["temperature"], ZS_CFG["prior_alpha"], ZS_CFG.get("cond_alpha", 0.0), ZS_CFG["ctx_events"])
    for ctx in ZS_CFG.get("ctx_events_grid", [ZS_CFG["ctx_events"]]):
        ZS_CFG["ctx_events"] = ctx
        for t in ZS_CFG["grid_taus"]:
            for a in ZS_CFG["grid_alphas"]:
                for c in ZS_CFG["grid_cond"]:
                    ZS_CFG["temperature"] = t
                    ZS_CFG["prior_alpha"] = a
                    ZS_CFG["cond_alpha"]  = c
                    preds = []
                    for _, r in val_df.iterrows():
                        p_days, _, _ = predict_time(r["prefix"], r["last_act"], r["elapsed_days"])
                        preds.append(p_days)
                    mae = mean_absolute_error(y_true, np.array(preds))
                    if mae < best[0]:
                        best = (mae, t, a, c, ctx)

    ZS_CFG["temperature"], ZS_CFG["prior_alpha"], ZS_CFG["cond_alpha"], ZS_CFG["ctx_events"] = best[1], best[2], best[3], best[4]
    wandb.config.update(
        {"zs_cfg_tuned": {"val_mae_days": float(best[0]), "temperature": best[1], "prior_alpha": best[2], "cond_alpha": best[3], "ctx_events": best[4]}},
        allow_val_change=True
    )
    log.info("Tuned on val → MAE=%.4f d, tau=%.2f, alpha=%.2f, cond=%.2f, ctx=%d", best[0], best[1], best[2], best[3], best[4])

tune_on_val()

# Record final knobs
wandb.config.update({
    "final_ctx_events": ZS_CFG["ctx_events"],
    "final_temperature": ZS_CFG["temperature"],
    "final_prior_alpha": ZS_CFG["prior_alpha"],
    "final_cond_alpha": ZS_CFG.get("cond_alpha", 0.0),
    "bin_mids_days": [float(x) for x in BIN_MIDS],
    "unit": config["unit"],
}, allow_val_change=True)

# %% Per-k evaluation
k_vals, counts, maes, mses, rmses = [], [], [], [], []
for k in sorted(test_df["k"].astype(int).unique()):
    subset = test_df[test_df["k"] == k]
    if subset.empty: continue
    y_true = subset["rt_days"].values.astype(np.float64)
    preds = []
    for _, r in subset.iterrows():
        p_days, _, _ = predict_time(r["prefix"], r["last_act"], r["elapsed_days"])
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

# (Optional) top-1 bin accuracy (coarse classification view)
y_true_bins = test_df["bin_idx"].values if len(test_df) else np.array([])
y_pred_bins = []
for _, r in test_df.iterrows():
    scores, _ = score_bins(r["prefix"], r["last_act"], r["elapsed_days"])  # add elapsed_days
    y_pred_bins.append(int(np.argmax(scores)))
if len(y_pred_bins):
    top1_bin_acc = accuracy_score(y_true_bins, y_pred_bins)
    print(f"Top-1 bin accuracy: {top1_bin_acc:.4f}")
    wandb.log({"metrics/top1_bin_acc": float(top1_bin_acc)})

# %% Plots → disk
plot_dir = config["plots_dir"]
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
    plt.title('RMSE vs. Prefix Length (k)'); plt.xlabel('Prefix Length (k)'); plt.ylabel('RMSE (days)')
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
sample = test_df.sample(n=min(5, len(test_df)), random_state=config["seed"]) if len(test_df) else test_df
tab = wandb.Table(columns=["k","prefix","last_act","gold_days","pred_days","top_bins","top_probs"])
for _, r in sample.iterrows():
    pred_days, top_bins, top_probs = predict_time(r["prefix"], r["last_act"], r["elapsed_days"], return_topk=3)
    print("Trace:", " → ".join(r["prefix"]))
    print(f"Last act: {r['last_act']}")
    print(f"Gold (days): {r['rt_days']:.4f}")
    print(f"Pred (days): {pred_days:.4f}")
    print("Top bins:", top_bins)
    print("-"*60)
    tab.add_data(
        r["k"],
        " → ".join(r["prefix"]),
        r["last_act"],
        float(r["rt_days"]),
        float(pred_days),
        ", ".join(top_bins),
        ", ".join([f"{p:.3f}" for p in top_probs])
    )
wandb.log({"samples": tab})

# %%
wandb.finish()