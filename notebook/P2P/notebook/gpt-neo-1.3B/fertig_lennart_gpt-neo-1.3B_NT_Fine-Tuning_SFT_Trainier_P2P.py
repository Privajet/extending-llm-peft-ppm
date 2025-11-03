# %% Fine-tuned Next-Time (NT) with GPT-Neo-1.3B — LoRA + SFT
# - Cast NT regression to classification over TRAIN-derived quantile bins (days).
# - Supervised fine-tuning with TRL SFT: completion-only loss after "Answer:" anchor.
# - PEFT LoRA on GPT-Neo attention/projection layers (c_attn, c_proj) for efficiency.
# - Deterministic setup: fixed seeds, gradient checkpointing, use_cache=False during train.
# - Tokenizer left-truncation; prompts list all time-bin labels; the gold target is the bin label text.
# - Inference: score each bin by log-likelihood of [prompt + bin_label], softmax to bin probs,
#   then decode numeric NT as expected value over bin centers; also return top-k bins.
# - Evaluation: MAE / MSE / RMSE vs. prefix length (k); plots saved and logged to W&B.
# - Artifacts: save LoRA adapters + tokenizer; reload via PEFT for evaluation-time scoring.

import os, sys, glob, ctypes, random, logging
os.environ["MPLBACKEND"]="Agg"
os.environ["TRANSFORMERS_NO_TORCHVISION"]="1"
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")

# Preload libstdc++ on some HPC stacks (no-op if not needed)
prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
cands = glob.glob(os.path.join(prefix, "lib", "libstdc++.so.6*"))
if cands:
    try:
        mode = getattr(os, "RTLD_GLOBAL", 0)
        ctypes.CDLL(cands[0], mode=mode)
    except OSError:
        pass

import numpy as np, pandas as pd, torch
from datetime import datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
from torch.nn.utils.rnn import pad_sequence

import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%  W&B
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key) if api_key else wandb.login()

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# %% 
DATASET = "P2P"

config = {
    # bookkeeping
    "dataset":                  DATASET,
    "plots_dir":                f"/ceph/lfertig/Thesis/notebook/{DATASET}/plots/gpt-neo-1.3B/FT/NT",
    "out_dir":                  f"/ceph/lfertig/Thesis/models/{DATASET}/gpt-neo-1.3B/NT/nt_ft_{ts}",
    "unit":                     "days"
}

FT_CFG = {
    # model / runtime
    "model_name":               "EleutherAI/gpt-neo-1.3B",
    "dtype":                    "fp16",                                 # set "fp32" if CPU-only
    "device":                   "auto",
    # prompt & context
    "max_seq_len":              512,
    "event_sep":                " → ",
    "prompt_tmpl_demo":         (
                                "Trace: {trace}\n"
                                "Predict the time until the next event. Choose EXACTLY ONE label from the list below and output ONLY that label.\n"
                                "Labels:\n{labels}\n"
                                "Answer: {gold}\n\n"
                                ),
    "prompt_tmpl_query":        (
                                "Trace: {trace}\n"
                                "Predict the time until the next event. Choose EXACTLY ONE label from the list below and output ONLY that label.\n"
                                "Labels:\n{labels}\n"
                                "Answer:"
                                ),
    "epochs":                   3,
    "micro_bsz":                1,
    "grad_accum":               8,
    "lr":                       3e-4,
    "warmup_ratio":             0.05,
    "lora_r":                   16, 
    "lora_alpha":               64, 
    "lora_dropout":             0.05,
    # time binning
    "n_time_bins":              20
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
    project=f"gpt-neo-1.3B_NT_FineTuning_{config['dataset']}",
    entity="privajet-university-of-mannheim",
    name=f"neo_ft_nt_{ts}",
    config=config,
    resume="never",
    force=True
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

# %% Time bins (TRAIN-only)
def make_bins(train_series, num_bins=20, min_bin_count=5, clip_low=1e-6, clip_high=None):
    def _quantile(x, qs):
        try:
            return np.quantile(x, qs, method="nearest")
        except TypeError:
            return np.quantile(x, qs, interpolation="nearest")

    x = train_series.values.astype(np.float64)
    if clip_low  is not None: x = np.maximum(x, clip_low)
    if clip_high is not None: x = np.minimum(x, clip_high)

    qs = np.linspace(0, 1, num_bins+1)
    edges = np.unique(_quantile(x, qs))
    if len(edges) < 3:
        lo, hi = x.min(), x.max()
        if hi <= lo: hi = lo + 1e-6
        edges = np.exp(np.linspace(np.log(max(lo, 1e-6)), np.log(hi), num_bins+1))

    # ensure a minimum count per bin by merging the smallest bins
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
            if i == 0:
                edges = np.delete(edges, i+1)
            elif i == len(counts)-1:
                edges = np.delete(edges, i)
            else:
                edges = np.delete(edges, i+1 if counts[i+1] <= counts[i-1] else i)
        return edges
    
    edges = enforce_min_count(edges, x, min_bin_count)
    centers = 0.5 * (edges[:-1] + edges[1:])
    labels = [f"({edges[i]:.5f}, {edges[i+1]:.5f}] {config['unit']}" for i in range(len(centers))]
    return np.array(edges, dtype=np.float64), np.array(centers, dtype=np.float64), labels

BIN_EDGES, BIN_CENTERS, BIN_LABELS = make_bins(
    train_df["nt_days"],
    num_bins=FT_CFG["n_time_bins"],
    min_bin_count=5,
    clip_low=1e-6,
    clip_high=None
)
n_bins = len(BIN_LABELS)

def digitize_nt(x):
    idx = np.digitize(float(x), BIN_EDGES, right=True) - 1
    return int(np.clip(idx, 0, n_bins-1))

for frame in (train_df, val_df, test_df):
    frame["bin_idx"]   = frame["nt_days"].apply(digitize_nt)
    frame["bin_label"] = frame["bin_idx"].apply(lambda i: BIN_LABELS[i])

# Prompt label list = time bins
labels_for_prompt = "\n".join(BIN_LABELS)

# %% Model / Tokenizer
MODEL_NAME = FT_CFG["model_name"]
DTYPE = (torch.float16 if (torch.cuda.is_available() and FT_CFG["dtype"]=="fp16") else torch.float32)
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and FT_CFG["device"]=="auto") else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

tokenizer_pad_id = tokenizer.pad_token_id
BIN_IDS = {lbl: tokenizer(" " + lbl, add_special_tokens=False).input_ids for lbl in BIN_LABELS}

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True).to(DEVICE)
if hasattr(base_model.config, "use_cache"): base_model.config.use_cache=False

print("PAD token:", tokenizer.pad_token, "PAD id:", tokenizer.pad_token_id)

# %%
target_modules = ["c_attn", "c_proj"]
peft_cfg = LoraConfig(
    r=FT_CFG["lora_r"], lora_alpha=FT_CFG["lora_alpha"], lora_dropout=FT_CFG["lora_dropout"],
    target_modules=target_modules, bias="none", task_type="CAUSAL_LM", inference_mode=False
)
model = get_peft_model(base_model, peft_cfg)
try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    model.gradient_checkpointing_enable()

# Make inputs require grad for checkpointing (prevents your error)
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    # Fallback hook for older transformers
    def _make_inputs_require_grad(module, inputs, output):
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

# keep cache off during train
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

model.print_trainable_parameters()

def df_to_text(df: pd.DataFrame, with_gold=True):
    sep = FT_CFG.get("event_sep", " → ")
    rows = []
    for _, r in df.iterrows():
        trace = sep.join(r["prefix"])
        if with_gold:
            rows.append(
                FT_CFG["prompt_tmpl_demo"].format(
                    trace=trace, labels=labels_for_prompt, gold=r["bin_label"]
                )
            )
        else:
            rows.append(
                FT_CFG["prompt_tmpl_query"].format(
                    trace=trace, labels=labels_for_prompt
                )
            )
    return rows

ds = DatasetDict({
    "train": Dataset.from_dict({"text": df_to_text(train_df, with_gold=True)}),
    "validation": Dataset.from_dict({"text": df_to_text(val_df, with_gold=True)}),
})

# Mask loss to completion only
class CompletionOnlyCollator:
    def __init__(self, tokenizer, anchor="Answer:", max_length=FT_CFG["max_seq_len"]):
        self.tok = tokenizer
        self.max_length = max_length
        self.anchor_ids = tokenizer(anchor, add_special_tokens=False).input_ids

    @staticmethod
    def _find_last(seq, pat):
        for i in range(len(seq)-len(pat), -1, -1):
            if seq[i:i+len(pat)] == pat: return i
        return -1

    def _mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()
        B, T = input_ids.size()
        arange_T = torch.arange(T)
        for i in range(B):
            ids = input_ids[i].tolist()
            j = self._find_last(ids, self.anchor_ids)
            labels[i].fill_(-100)
            if j != -1:
                cut = j + len(self.anchor_ids)
                keep = (attention_mask[i] == 1) & (arange_T >= cut)
                labels[i, keep] = input_ids[i, keep]
        return labels

    def __call__(self, feats):
        # feats are already tokenized by TRL (have input_ids / attention_mask)
        batch = self.tok.pad(feats, padding=True, return_tensors="pt")
        if batch["input_ids"].size(1) > self.max_length:
            batch["input_ids"]      = batch["input_ids"][:, :self.max_length]
            batch["attention_mask"] = batch["attention_mask"][:, :self.max_length]
        batch["labels"] = self._mask(batch["input_ids"], batch["attention_mask"])
        return batch

collator = CompletionOnlyCollator(tokenizer)

# %%
sft_cfg = SFTConfig(
    output_dir=config["out_dir"],
    num_train_epochs=FT_CFG["epochs"],
    learning_rate=FT_CFG["lr"],
    per_device_train_batch_size=FT_CFG["micro_bsz"],
    per_device_eval_batch_size=FT_CFG["micro_bsz"],
    gradient_accumulation_steps=FT_CFG["grad_accum"],
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=FT_CFG["warmup_ratio"],
    weight_decay=0.05,
    optim="adamw_torch",
    max_grad_norm=0.5,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    bf16=False,
    fp16=True,
    max_seq_length=FT_CFG["max_seq_len"],
    packing=False,
    dataset_text_field="text",
    report_to=["wandb"],
)

trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
)

# %% Train & save
trainer.train()
save_dir=config["out_dir"]
trainer.model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)
log.info("Saved adapters & tokenizer to %s", save_dir)

# %% Load for inference
gen_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, low_cpu_mem_usage=True)
gen_model.resize_token_embeddings(len(tokenizer)); gen_model.to(DEVICE)
gen_model = PeftModel.from_pretrained(gen_model, save_dir)
if hasattr(gen_model,"gradient_checkpointing_disable"): gen_model.gradient_checkpointing_disable()
gen_model.eval()
if hasattr(gen_model.config,"use_cache"): gen_model.config.use_cache=True
device = next(gen_model.parameters()).device

# %% Scoring & Evaluation
def _scores_for_bins(prefix, labels=BIN_LABELS):
    """Return per-bin log-likelihood scores for continuing the prompt with each bin label."""
    sep = FT_CFG.get("event_sep", " → ")
    seq = sep.join(prefix)
    prompt = (FT_CFG["prompt_tmpl_query"].format(trace=seq, labels=labels_for_prompt)) + " "
    P_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)

    rows, lens = [], []
    for lb in labels:
        L_ids = torch.tensor(BIN_IDS[lb], dtype=torch.long, device=device)
        rows.append(torch.cat([P_ids, L_ids], dim=0))
        lens.append(int(L_ids.size(0)))

    input_ids = pad_sequence(rows, batch_first=True, padding_value=tokenizer_pad_id).to(device)
    attention_mask = (input_ids != tokenizer_pad_id).long()

    with torch.no_grad():
        logits = gen_model(input_ids=input_ids, attention_mask=attention_mask).logits

    cut = P_ids.size(0)
    scores = []
    for i, lb in enumerate(labels):
        L = lens[i]
        lp  = torch.log_softmax(logits[i, cut-1:cut-1+L, :], dim=-1)
        tgt = torch.tensor(BIN_IDS[lb], device=device)
        scores.append(float(lp.gather(-1, tgt.unsqueeze(-1)).sum()))
    return np.array(scores, dtype=np.float32)

def _softmax_np(x: np.ndarray):
    x = x.astype(np.float32); x -= x.max()
    e = np.exp(x); s = e.sum()
    return e/s if s > 0 else np.ones_like(x)/len(x)

def predict_time(prefix, last_act=None, return_topk=3):
    """Return (pred_days, top_bins, top_probs). Numeric pred is expected value over BIN_CENTERS."""
    scores = _scores_for_bins(prefix, labels=BIN_LABELS)
    probs  = _softmax_np(scores)
    pred_days = float((probs * np.asarray(BIN_CENTERS, dtype=np.float32)).sum())

    k = min(return_topk, len(BIN_LABELS))
    idx = np.argsort(probs)[-k:][::-1]
    top_bins  = [BIN_LABELS[i] for i in idx]
    top_probs = [float(probs[i]) for i in idx]
    return pred_days, top_bins, top_probs

# %% Per-k evaluation (days)
k_vals, counts, maes, mses, rmses = [], [], [], [], []
for k in sorted(test_df["k"].astype(int).unique()):
    subset = test_df[test_df["k"] == k]
    if subset.empty:
        continue
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

# %% Optional: bin top-k hit rate
def topk_bin_hit_rate(df, k=3):
    hits, N = 0, len(df)
    for _, r in df.iterrows():
        _, top_bins, _ = predict_time(r["prefix"], r["last_act"], return_topk=k)
        if r["bin_label"] in top_bins:
            hits += 1
    return hits / N if N else float("nan")

wandb.log({
    "metrics/bin_top3_hit": float(topk_bin_hit_rate(test_df, 3)),
    "metrics/bin_top5_hit": float(topk_bin_hit_rate(test_df, 5)),
})

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
sample = test_df.sample(n=min(5, len(test_df)), random_state=config["seed"]) if len(test_df) else test_df
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