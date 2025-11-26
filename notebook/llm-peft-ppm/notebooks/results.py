# %%
import os
import sys
import warnings
from pprint import pprint

import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from peft import LoraConfig, TaskType

from ppm.wandb_utils import fetch_experiments
from ppm.models import NextEventPredictor
from ppm.models.config import FreezeConfig

warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

if project_root not in sys.path:
    sys.path.append(project_root)

entity = os.environ.get("ENTITY")
project = os.environ.get("PROJECT")

print("Current working directory:", os.getcwd())
print("project_root in sys.path:", project_root in sys.path)

# %% 
output_dir_csv = "/ceph/lfertig/Thesis/notebook/llm-peft-ppm/notebooks/csv"
output_dir_plots = "/ceph/lfertig/Thesis/notebook/llm-peft-ppm/notebooks/plots"
os.makedirs(output_dir_csv, exist_ok=True)
os.makedirs(output_dir_plots, exist_ok=True)

# %%
# - display all lines pandas
pd.set_option("display.max_rows", None)

mpl.rcParams.update({
    "figure.figsize": (6, 4),          
    "font.size": 10,                   
    "axes.labelsize": 10,              
    "axes.titlesize": 10,              
    "legend.fontsize": 9,              
    "xtick.labelsize": 9,              
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,            
    "lines.markersize": 5,             
    "axes.grid": True,                 
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "legend.frameon": False,           
    "pdf.fonttype": 42,                
    "ps.fonttype": 42,
    "savefig.bbox": "tight",           
    "savefig.dpi": 300,                
})

colors = [
    "#9467bd",
    "#2ca02c",
    "#bcbd22",
    "#7f7f7f",
    "#e377c2",
    "#8c564b",
    "#d62728",
    "#17becf",
    "#1f77b4",
    "#ff7f0e",
]

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)  

# %% Experimental setup
# - datasets' characteristics
# - Architecture illustration

# Datasets
from skpm.event_logs import (
    BPI12,
    BPI17,
    BPI20PrepaidTravelCosts,
    BPI20TravelPermitData,
    BPI20RequestForPayment,
)

logs = [
    BPI12,
    BPI17,
    BPI20PrepaidTravelCosts,
    BPI20TravelPermitData,
    BPI20RequestForPayment,
]

properties = pd.DataFrame()
for log in logs:
    df = log().dataframe

    p = {
        "Log": log.__name__,
        "# cases": len(df["case:concept:name"].unique()),
        "# evt.": len(df),
        "# act.": len(df["concept:name"].unique()),
        "tlmean": df.groupby("case:concept:name").size().mean(),
        "tlstd": df.groupby("case:concept:name").size().std(),
    }

    p = pd.DataFrame(p, index=[0])
    properties = pd.concat([properties, p], ignore_index=True)

# %% Datasets' characteristics
properties['Trace length'] = properties.apply(
    lambda row: f"{row['tlmean']:.4f}±{row['tlstd']:.1f}",
    axis=1
)
properties = properties.drop(columns=["tlmean", "tlstd"])
log_props_path = os.path.join(output_dir_csv, "log_properties.csv")
properties.sort_values(by="# evt.").round(4).to_csv(log_props_path, index=False)
print("Log properties:")
print(properties.sort_values(by="# evt.").round(4))

# %% Architecture illustration

rnn_example = NextEventPredictor(
    embedding_size=32,
    categorical_cols=["activity"],
    numerical_cols=["accumulated_time"],
    categorical_sizes={
        "activity": 20,
    },
    categorical_targets=["activity"],
    numerical_targets=["remaining_time"],
    backbone_name="rnn",
    backbone_hidden_size=64,
    backbone_n_layers=2,
    padding_idx=0,
    strategy="sum",
    backbone_pretrained=False,
    backbone_finetuning=None,
    backbone_type="lstm",
    device="cuda",
)
pprint(rnn_example)

fine_tuning = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=2,
    lora_alpha=4,
    use_rslora=True,
)

# NOTE: you must set the enviroment variable HF_TOKEN on your local machine to access models from the huggingface hub
qwen_input_size = 896
qwen_example = NextEventPredictor(
    embedding_size=qwen_input_size,
    categorical_cols=["activity"],
    numerical_cols=["accumulated_time"],
    categorical_sizes={
        "activity": 20,
    },
    categorical_targets=["activity"],
    numerical_targets=["remaining_time", "time_to_next_event"],
    backbone_name="Qwen/Qwen2.5-0.5B",
    backbone_pretrained=True,
    backbone_hidden_size=qwen_input_size,
    backbone_finetuning=fine_tuning,

    backbone_type=None,
    backbone_n_layers=None,
    padding_idx=0,
    strategy="sum",
    device="cuda",
)
pprint(qwen_example)
del qwen_example

# %%
freeze_config = FreezeConfig(
    ix_layers=[0, 1, 2],
    module_path="layers",
)

# NOTE: you must set the enviroment variable HF_TOKEN on your local machine to access models from the huggingface hub
llama_input_size = 2048
llama_example = NextEventPredictor(
    embedding_size=llama_input_size,
    categorical_cols=["activity"],
    numerical_cols=["accumulated_time"],
    categorical_sizes={
        "activity": 20,
    },
    categorical_targets=["activity"],
    numerical_targets=["remaining_time", "time_to_next_event"],
    backbone_name="unsloth/Llama-3.2-1B",
    backbone_pretrained=True,
    backbone_hidden_size=llama_input_size,
    backbone_finetuning=freeze_config,

    backbone_type=None,
    backbone_n_layers=None,
    padding_idx=0,
    strategy="sum",
    device="cuda",
)
pprint(llama_example)

# %% Illustrating how the freezing fine-tuning works.
# In the `freeze_config` we pass as argument a list of `ix_layers` indicating which layer must be fine-tuned. In this case, we are fine-tuning the three first layers. All the other layers are frozen.

LAYER_TO_TUNE = 0
LAYER_TO_FREEZE = 3

print("Tuning layer[0]:")    
for name, param in llama_example.backbone.layers[LAYER_TO_TUNE].named_parameters():
    print(name, param.requires_grad)

print("-"*80 + "\nFreezing layer[3]:")    
for name, param in llama_example.backbone.layers[LAYER_TO_FREEZE].named_parameters():
    print(name, param.requires_grad)

del llama_example

# %%
def map_setting(row):
    ft = row["fine_tuning"]
    k  = row.get("few_shot_k", None)
    fl = row.get("freeze_layers", None)
    epochs = row.get("epochs", None)

    # LoRA Few-Shot
    if ft == "lora" and k == 8:
        return "FewShot-LoRA"

    # LoRA Full
    if ft == "lora" and pd.isna(k):
        return "LoRA"

    # Zero-Shot
    if ft == "freeze" and epochs == 0:
        return "ZeroShot"

    # Freezing Few-Shot
    if ft == "freeze" and k == 8:
        return "FewShot-Freezing"

    # Freezing standard
    if ft == "freeze" and pd.isna(fl):
        return "Freezing"

    # Freezing layer configs
    if ft == "freeze" and isinstance(fl, str):
        fl_clean = fl.replace("[","").replace("]","").split()
        return "Freezing-" + str(fl_clean)

    return "Other"

# %% Checking best models

BACKBONE_PROJECTS = {
    "majority":         "llm-peft-ppm_majority_baseline",
    "rnn":              "llm-peft-ppm_rnn",
    "transformer":      "llm-peft-ppm_transformer_baseline",
    "tabpfn":           "llm-peft-ppm_tabpfn_baseline",
    "gpt2":             "llm-peft-ppm_gpt2",
    "gptneo-1b3":       "llm-peft-ppm_gpt-neo-1.3B",
    "qwen25-05b":       "llm-peft-ppm_qwen25-05b",
    "llama32-1b":       "llm-peft-ppm_llama32-1b",
    "gemma-2-2b":       "llm-peft-ppm_gemma-2-2b",
}

all_results = []

for backbone, project_name in BACKBONE_PROJECTS.items():
    df = fetch_experiments(project=project_name, entity=entity, include_metrics=True)
    df["backbone"] = backbone  # force consistent naming
    df["project"] = project_name
    all_results.append(df)

global_results = pd.concat(all_results, ignore_index=True)

cols = [
    "id",
    "log",
    "backbone",
    "project",
    "fine_tuning",
    "total_params",
    "trainable_params",
    "test_next_activity_acc",
    "test_next_activity_loss",
    "test_next_remaining_time_loss",
    "test_next_time_to_next_event_loss",
    "best_test_next_activity_acc",
    "best_test_next_activity_loss",
    "best_test_next_remaining_time_loss",
    "best_test_next_time_to_next_event_loss",
    "_runtime",
    "mt_score",
]

global_results_eval = global_results[
    (~global_results["test_next_activity_acc"].isna()) &
    (~global_results["test_next_remaining_time_loss"].isna()) &
    (~global_results["test_next_time_to_next_event_loss"].isna())
].copy()

# normalize metrics and build equal-weight multi-task score
sc_acc = MinMaxScaler()
sc_rt  = MinMaxScaler()
sc_nt  = MinMaxScaler()

global_results_eval["na_norm"] = sc_acc.fit_transform(
    global_results_eval[["test_next_activity_acc"]]
)
# losses: lower is better -> negate before scaling
global_results_eval["rt_norm"] = sc_rt.fit_transform(
    -global_results_eval[["test_next_remaining_time_loss"]]
)
global_results_eval["nt_norm"] = sc_nt.fit_transform(
    -global_results_eval[["test_next_time_to_next_event_loss"]]
)

global_results_eval["mt_score"] = (
    global_results_eval["na_norm"] +
    global_results_eval["rt_norm"] +
    global_results_eval["nt_norm"]
)

best_scores = (
    global_results_eval
    .groupby(["log", "backbone"])["mt_score"]
    .max()
    .dropna()
)

best = global_results_eval.merge(best_scores, on=["log","backbone","mt_score"])
best = best[cols].reset_index(drop=True)

for log_name, df_log in best.groupby("log"):
    csv_path = os.path.join(
        output_dir_csv,
        f"best_runs_mt_score_{log_name}.csv"
    )
    df_log.to_csv(csv_path, index=False)
    print(f"Saved best models for log={log_name} to {csv_path}")
    
# %% Multi-task models
METRICS = [
    "best_test_next_activity_acc",
    "best_test_next_remaining_time_loss",
    "best_test_next_time_to_next_event_loss",
]

# derive high-level setting labels (LoRA, FewShot-LoRA, Freezing, etc.)
best = best.copy()
best["Setting"] = best.apply(map_setting, axis=1)

tmp = best.reset_index(drop=True).copy()
tmp = tmp[
    ["log", "backbone", "Setting", "fine_tuning"]
    + METRICS
    + ["total_params", "trainable_params", "_runtime"]
]

# seconds to hours
tmp["_runtime"] = (tmp["_runtime"] / 3600.0).round(3)

# params formatting
tmp["trainable_params"] = (
    (tmp.trainable_params / tmp.total_params) * 100
).astype(int).astype(str) + "%"
tmp["total_params"] = tmp["total_params"].apply(
    lambda x: np.format_float_scientific(x, precision=1)
)
tmp["# params\n(%trainable)"] = (
    tmp["total_params"] + "(" + tmp["trainable_params"] + ")"
)

data = tmp.rename(
    columns={
        "log":                              "Dataset",
        "backbone":                         "Backbone",
        "best_test_next_activity_acc":      "NA Acc.",
        "best_test_next_remaining_time_loss": "RT MSE",
        "best_test_next_time_to_next_event_loss": "NT MSE",
        "total_params":                     "# params\n(total)",
        "trainable_params":                 "% params\n(trainable)",
        "_runtime":                         "Runtime (h)",
    }
)

# pretty dataset names (optional – you already use this)
data.Dataset = data.Dataset.map(
    {
        "BPI12":                            "BPI12",
        "BPI17":                            "BPI17",
        "BPI20PrepaidTravelCosts":          "BPI20PTC",
        "BPI20RequestForPayment":           "BPI20RfP",
        "BPI20TravelPermitData":            "BPI20TPD",
    }
)

# pretty backbone names (no [LoRA] etc!)
data.Backbone = data.Backbone.map(
    {
        "majority":                         "Majority",
        "rnn":                              "RNN",
        "transformer":                      "Transformer",
        "tabpfn":                           "TabPFN",
        "gpt2":                             "GPT2",
        "gptneo-1b3":                       "GPT-Neo-1b3",
        "qwen25-05b":                       "Qwen2.5-0.5b",
        "llama32-1b":                       "Llama3.2-1b",
        "gemma-2-2b":                       "Gemma-2-2b",
    }
)

# you can optionally order Setting explicitly, e.g.:
SETTING_ORDER = [
    "ZeroShot",
    "LoRA",
    "FewShot-LoRA",
    "Freezing",
    "FewShot-Freezing",
    "Freezing-[-1]",
    "Freezing-[-1, -2]",
    "Freezing-[0]",
    "Freezing-[0, 1]",
    "Other",
]
data["Setting"] = pd.Categorical(data["Setting"], categories=SETTING_ORDER, ordered=True)

# final sort: by Dataset, Backbone, then Setting
data = data.sort_values(by=["Dataset", "Backbone", "Setting"]).reset_index(drop=True)

csv_path = os.path.join(output_dir_csv, "multi-task_benchmark_results.csv")
data.to_csv(csv_path, index=False)

# %%  Experimental evaluation
global_multi_task_results = global_results[
    (~global_results["test_next_remaining_time_loss"].isna()) &
    (~global_results["test_next_time_to_next_event_loss"].isna()) &
    (~global_results["test_next_activity_loss"].isna())
].copy()

BACKBONE_LABELS = {
    "gpt2":                                 "GPT2",
    "gptneo-1b3":                           "GPT-Neo-1.3B",
    "qwen25-05b":                           "Qwen-2.5-0.5B",
    "llama32-1b":                           "Llama-3.2-1B",
    "gemma-2-2b":                           "Gemma-2-2B",
}
global_multi_task_results["Backbone"] = global_multi_task_results["backbone"].map(BACKBONE_LABELS)

logs_to_plot = sorted(global_multi_task_results["log"].unique())

for log_name in logs_to_plot:
    subset = global_multi_task_results[global_multi_task_results["log"] == log_name]

    if subset.empty:
        continue

    fig, ax = plt.subplots(3, 1, figsize=(6, 6), sharex=True, dpi=100)

    sns.boxplot(
        data=subset,
        x="Backbone",
        y="test_next_activity_acc",
        ax=ax[0],
    )

    sns.boxplot(
        data=subset,
        x="Backbone",
        y="test_next_remaining_time_loss",
        ax=ax[1],
    )

    sns.boxplot(
        data=subset,
        x="Backbone",
        y="test_next_time_to_next_event_loss",
        ax=ax[2],
    )

    ax[0].set_ylabel("NA Acc.")
    ax[1].set_ylabel("RT MSE")
    ax[2].set_ylabel("NT MSE")

    for a in ax:
        a.set_xlabel("")
        a.set_title(log_name)

    plt.tight_layout()

    pdf_path = os.path.join(output_dir_plots, f"loss_distribution_{log_name}.pdf")
    png_path = os.path.join(output_dir_plots, f"loss_distribution_{log_name}.png")
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)

    plt.close(fig)

# %%
global_results_multi = global_results.copy()
global_results_multi["Setting"] = global_results_multi.apply(map_setting, axis=1)

SETTING_ORDER = [
    "ZeroShot",
    "LoRA",
    "FewShot-LoRA",
    "Freezing",
    "FewShot-Freezing",
    "Freezing-[0]",
    "Freezing-[0,1]",
    "Freezing-[-1]",
    "Freezing-[-1,-2]",
]

for backbone in sorted(global_results_multi["backbone"].unique()):
    subset = global_results_multi[global_results_multi["backbone"] == backbone]

    subset = subset[subset["Setting"].notna()].copy()
    if subset.empty:
        continue
        
    setting_order_current = [
        s for s in SETTING_ORDER if s in subset["Setting"].unique()
    ]
    if not setting_order_current:
        continue

    fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True, dpi=100)

    sns.boxplot(
        data=subset,
        x="Setting",
        y="test_next_activity_acc",
        order=setting_order_current,
        ax=ax[0],
    )

    sns.boxplot(
        data=subset,
        x="Setting",
        y="test_next_remaining_time_loss",
        order=setting_order_current,
        ax=ax[1],
    )

    sns.boxplot(
        data=subset,
        x="Setting",
        y="test_next_time_to_next_event_loss",
        order=setting_order_current,
        ax=ax[2],
    )

    ax[0].set_ylabel("NA Acc.")
    ax[1].set_ylabel("RT MSE")
    ax[2].set_ylabel("NT MSE")
    ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, ha='right')

    fig.suptitle(f"Backbone: {backbone}", fontsize=12)
    plt.tight_layout()

    path = os.path.join(output_dir_plots, f"loss_distribution_settings_comparison_{backbone}.pdf")
    plt.savefig(path, dpi=300)
    plt.close(fig)

# %%
param_summary = best[["log", "total_params", "trainable_params"]].copy()

param_summary["trainable_percent"] = (
    (param_summary["trainable_params"] / param_summary["total_params"]) * 100
).astype(int).astype(str) + "%"

param_summary["total_params_fmt"] = param_summary["total_params"].apply(
    lambda x: np.format_float_scientific(x, precision=1)
)

param_summary["# params\n(%trainable)"] = (
    param_summary["total_params_fmt"] + "(" + param_summary["trainable_percent"] + ")"
)

csv_path = os.path.join(output_dir_csv, "param_summary.csv")
param_summary.to_csv(csv_path, index=False)

print("=== PARAMETER SUMMARY PER DATASET ===")
print(param_summary.to_string(index=False))

# %% Loss curves
# multi-task models

def fetch_single(
    wandb_id: str,
    targets=["na", "rt", "nt"], 
    project_name: str | None = None,
    entity: str | None = None,
):
    if isinstance(targets, str):
        targets = [targets]

    if project_name is None:
        raise ValueError("fetch_single requires an explicit project_name (model project).")
    entity = os.environ["ENTITY"] if entity is None else entity
        
    api = wandb.Api()
    run = api.run(f"{entity}/{project_name}/{wandb_id}")
    history = list(run.scan_history())

    na_acc, na_loss, rt_loss, nt_loss = None, None, None, None

    if "rt" in targets:
        rt_loss = [row["test_next_remaining_time_loss"] 
                   for row in history 
                   if "test_next_remaining_time_loss" in row]

    if "na" in targets:
        na_loss = [row["test_next_activity_loss"] 
                   for row in history 
                   if "test_next_activity_loss" in row]
        na_acc = [row["test_next_activity_acc"] 
                  for row in history 
                  if "test_next_activity_acc" in row]

    if "nt" in targets:
        nt_loss = [row["test_next_time_to_next_event_loss"] 
                   for row in history 
                   if "test_next_time_to_next_event_loss" in row]

    return na_acc, na_loss, rt_loss, nt_loss

# %%
loss_csv_path = os.path.join(output_dir_csv, "final_loss_curves_multitask.csv")

if os.path.exists(loss_csv_path):
    losses = pd.read_csv(loss_csv_path)
else:
    losses_list = []
    
    # iterate over best multi-task runs (one per log × backbone)
    for _, row in best.iterrows():
        na_acc, na_loss, rt_loss, nt_loss = fetch_single(
            row.id,
            project_name=row.project,
            targets=["na", "rt", "nt"],
        )
        
        # build per-epoch dataframe for this run
        tmp = pd.DataFrame({
            "epoch": range(len(na_loss)),
            "na_acc": na_acc,
            "na_loss": na_loss,
            "rt_loss": rt_loss,
            "nt_loss": nt_loss,
        })
        tmp["log"] = row.log
        tmp["backbone"] = row.backbone

        losses_list.append(tmp)
    
    losses = pd.concat(losses_list, axis=0, ignore_index=True)
    
    losses.to_csv(loss_csv_path, index=False)
    
# print a small sample into the log
print("=== MULTI-TASK LOSS CURVES (head) ===")
print(losses.head().to_string(index=False))

# %% Loss curve visualization (multi-task)
LOGS_TO_PLOT = sorted(losses["log"].unique())

HUE_MAP = {
    "gpt2":         "GPT2",
    "gptneo-1b3":   "GPT-Neo-1b3",
    "qwen25-05b":   "Qwen2.5-0.5b",
    "llama32-1b":   "Llama3.2-1b",
    "gemma-2-2b":   "Gemma-2-2b",
}

HUE_ORDER = [
    "GPT2",
    "GPT-Neo-1b3",
    "Qwen2.5-0.5b",
    "Llama3.2-1b",
    "Gemma-2-2b",
]

# reshape into tidy format
l = losses.melt(
    id_vars=["log", "backbone", "epoch"],
    value_vars=["na_loss", "rt_loss", "nt_loss"],
    var_name="Loss",
    value_name="Value",
).dropna(subset=["Value"])

# map model names to readable labels
l["Backbone"] = l["backbone"].map(HUE_MAP)
l = l[l["Backbone"].notna()]

LOSS_LABELS = {
    "na_loss": "NA Loss",
    "rt_loss": "RT Loss",
    "nt_loss": "NT Loss",
}

# prepare grid
fig, axes = plt.subplots(3, len(LOGS_TO_PLOT), figsize=(4 * len(LOGS_TO_PLOT), 8), sharex=True)
axes_iter = iter(axes.flatten())

for loss_name in ["na_loss", "rt_loss", "nt_loss"]:
    for log_name in LOGS_TO_PLOT:
        ax = next(axes_iter)
        tmp = l[(l["Loss"] == loss_name) & (l["log"] == log_name)]

        sns.lineplot(
            data=tmp,
            x="epoch",
            y="Value",
            hue="Backbone",
            hue_order=[h for h in HUE_ORDER if h in tmp["Backbone"].unique()],
            ax=ax,
            linewidth=2.0,
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(LOSS_LABELS[loss_name])
        ax.set_title(log_name)

        # legend handling
        if (loss_name, log_name) != ("nt_loss", LOGS_TO_PLOT[-1]):
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            leg = ax.legend(title="", ncol=3, bbox_to_anchor=(1.05, -0.3))
            for line in leg.get_lines():
                line.set_linewidth(2.0)

plt.tight_layout()

plot_path = os.path.join(output_dir_plots, "loss_curves_multitask.png")
plt.savefig(plot_path, dpi=300)
plt.close(fig)