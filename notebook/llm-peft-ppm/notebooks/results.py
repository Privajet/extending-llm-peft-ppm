# %%
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

if project_root not in sys.path:
    sys.path.append(project_root)

print("Current working directory:", os.getcwd())
print("project_root in sys.path:", project_root in sys.path)


# %%
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# plt.style.use("ggplot")

import matplotlib as mpl

# display all lines pandas
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

from ppm.wandb_utils import fetch_experiments

entity = os.environ.get("ENTITY")
project = os.environ.get("PROJECT")

results_dir = os.path.join("notebooks", f"{project}.csv")

if os.path.exists(results_dir):
    results = pd.read_csv(results_dir)
else:
    results = fetch_experiments(
        project=project,
        entity=entity,
        include_metrics=True
    )
    results.to_csv(results_dir, index=False)

print("Results by log/backbone:")
print(results.groupby(["log", "backbone"]).size())


# %%
def fetch_single(
    wandb_id: str,
    targets=["na", "rt", "nt"], 
    project_name: str | None = None,
    entity: str | None = None,
):
    if isinstance(targets, str):
        targets = [targets]

    project_name = os.environ["PROJECT"] if project_name is None else project_name
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
# Experimental setup
# - datasets' characteristics
# - architectures and illustrations
# - param count

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


# %%
properties['Trace length'] = properties.apply(
    lambda row: f"{row['tlmean']:.4f}±{row['tlstd']:.1f}",
    axis=1
)
properties = properties.drop(columns=["tlmean", "tlstd"])
properties.sort_values(by="# evt.").round(4).to_csv("notebooks/log_properties.csv", index=False)
print("Log properties:")
print(properties.sort_values(by="# evt.").round(4))

# %%
# Architecture illustration
# supress warning
import warnings
warnings.filterwarnings("ignore")

from pprint import pprint
from ppm.models import NextEventPredictor

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

from peft import LoraConfig, TaskType

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
from ppm.models.config import FreezeConfig

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

# %%
# check frozen layers
# Illustrating how the freezing fine-tuning works.
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
# Experimental evaluation
multi_task_results = results[
    (~results["test_next_remaining_time_loss"].isna()) &
    (~results["test_next_time_to_next_event_loss"].isna()) &
    (~results["test_next_activity_loss"].isna()) &
    (results.log.isin(["BPI12", "BPI17"])) & # just to smooth the plot
    (results.backbone.str.startswith(('qwen', 'llama', 'gpt2'))) &
    ((results["r"].isin([128, 256])) | (results["r"].isna()))
].copy()

backbone_mapping = {
    "llama32-1b": "Llama3.2",
    "qwen25-05b": "Qwen2.5",
    "gpt2": "GPT2",
}
multi_task_results["backbone"] = multi_task_results["backbone"].map(backbone_mapping)
multi_task_results["fine_tuning"] = multi_task_results["fine_tuning"].map({
    "lora": "LoRA",
    "freeze": "Freezing",
})

fig, ax = plt.subplots(3, 1, figsize=(4, 4), sharex=True, dpi=100)

order = ["GPT2", "Qwen2.5", "Llama3.2"]

# NA
sns.boxplot(
    data=multi_task_results,
    x="backbone",
    y="test_next_activity_acc",
    hue="fine_tuning",
    ax=ax[0],
    order=order,
)

# RT
sns.boxplot(
    data=multi_task_results,
    x="backbone",
    y="test_next_remaining_time_loss",
    hue="fine_tuning",
    ax=ax[1],
    order=order,
)

# NT
sns.boxplot(
    data=multi_task_results,
    x="backbone",
    y="test_next_time_to_next_event_loss",
    hue="fine_tuning",
    ax=ax[2],
    order=order,
)

ax[0].set_ylabel("NA Acc.")
ax[1].set_ylabel("RT MSE")
ax[2].set_ylabel("NT MSE")

for a in ax:
    a.set_title("")
    a.set_xlabel("")

ax[0].legend().remove()
ax[1].legend().remove()
ax[2].legend(title="", ncol=2, loc="upper left")

os.makedirs("notebooks/plots", exist_ok=True)
plot_path = os.path.join("notebooks", "plots", "loss_distribution_1.pdf")
plt.tight_layout()
plt.savefig(plot_path, dpi=300)

run = wandb.init(
    project=project,
    entity=entity,
    job_type="analysis",
    name="loss_distribution_1_plots",
)

wandb.log({"loss_distribution_1": wandb.Image(fig)})
run.finish()

# %%
import numpy as np
LOGS_TO_PLOT = ["BPI20PrepaidTravelCosts", "BPI12", "BPI17"]
HUE_ORDER=["Freezing", "Freezing-[0]", "Freezing-[0,1]", "Freezing-[-1]", "Freezing-[-1,-2]"]
HUE_MAP = {
    "gpt2": "GPT2", 
    "qwen25-05b": "Qwen2.5", 
    "llama32-1b": "Llama3.2"
}
ORDER = ["GPT2", "Qwen2.5", "Llama3.2"]
peft = results[
    (results.fine_tuning.isin(("freeze",)))
    # & (results.log.isin(LOGS_TO_PLOT))
    # & (results.r != 128)
    & (results.strategy != "concat")
].reset_index(drop=True).copy()
peft.backbone = peft.backbone.map(HUE_MAP)
peft.fine_tuning = peft.fine_tuning.map({
    "lora": "LoRA",
    "freeze": "Freezing",
})
peft.fine_tuning = np.where(peft.freeze_layers.isna(), peft.fine_tuning, peft.fine_tuning + "-[" + peft.freeze_layers.astype(str) + "]")

fig, ax = plt.subplots(3, 1, figsize=(5, 4), dpi=100, sharex=True)

# NA
sns.boxplot(
    x="backbone",
    y="test_next_activity_acc",
    hue="fine_tuning",
    hue_order=HUE_ORDER,
    order=ORDER,
    data=peft,
    ax=ax[0],
)

# RT
sns.boxplot(
    x="backbone",
    y="test_next_remaining_time_loss",
    hue="fine_tuning",
    hue_order=HUE_ORDER,
    order=ORDER,
    data=peft,
    ax=ax[1],
)

# NT
sns.boxplot(
    x="backbone",
    y="test_next_time_to_next_event_loss",
    hue="fine_tuning",
    hue_order=HUE_ORDER,
    order=ORDER,
    data=peft,
    ax=ax[2],
)

ax[0].set_xlabel("")
ax[0].set_ylabel("NA Acc.")
ax[1].set_xlabel("")
ax[1].set_ylabel("RT MSE")
ax[2].set_xlabel("")
ax[2].set_ylabel("NT MSE")

ax[0].legend().remove()
ax[1].legend().remove()
ax[2].legend(title="", ncol=2, loc="upper left")

os.makedirs("notebooks/plots", exist_ok=True)
plot_path = os.path.join("notebooks", "plots", "loss_distribution_2.pdf")
plt.tight_layout()
plt.savefig(plot_path, dpi=300)

run = wandb.init(
    project=project,
    entity=entity,
    job_type="analysis",
    name="loss_distribution_2_plots",
)

wandb.log({"loss_distribution_2": wandb.Image(fig)})
run.finish()

# %%
# ## Checking best models

BACKBONE_PROJECTS = {
    "majority": "llm-peft-ppm_majority_baseline",
    "rnn": "llm-peft-ppm_rnn",
    "transformer": "llm-peft-ppm_transformer_baseline",
    "tabpfn": "llm-peft-ppm_tabpfn_baseline",
    "gpt2": "llm-peft-ppm_gpt2",
    "llama32-1b": "llm-peft-ppm_llama32-1b",
    "qwen25-05b": "llm-peft-ppm_qwen25-05b",
    "gptneo-1b3": "llm-peft-ppm_gpt-neo-1.3B",
    "gemma-2-2b": "llm-peft-ppm_gemma-2-2b",
}

all_results = []

for backbone, project_name in BACKBONE_PROJECTS.items():
    df = fetch_experiments(project=project_name, entity=entity, include_metrics=True)
    df["backbone"] = backbone  # force consistent naming
    df["project"] = project_name
    all_results.append(df)

results = pd.concat(all_results, ignore_index=True)

cols = [
    "id", 
    "log", 
    "backbone", 
    "test_next_activity_acc", 
    "test_next_activity_loss", 
    "test_next_remaining_time_loss", 
    "test_next_time_to_next_event_loss",
    "project", 
    "best_test_next_activity_acc", 
    "best_test_next_activity_loss", 
    "best_test_next_remaining_time_loss",
    "best_test_next_time_to_next_event_loss",
    "trainable_params",
    "total_params",
    "fine_tuning",
    "_runtime"
]

results_eval = results[
    (~results["test_next_activity_loss"].isna()) &
    (~results["test_next_remaining_time_loss"].isna()) &
    (~results["test_next_time_to_next_event_loss"].isna())
].copy()

# normalize metrics and build equal-weight multi-task score
sc_acc = MinMaxScaler()
sc_rt  = MinMaxScaler()
sc_nt  = MinMaxScaler()

results_eval["na_norm"] = sc_acc.fit_transform(
    results_eval[["test_next_activity_acc"]]
)
# losses: lower is better -> negate before scaling
results_eval["rt_norm"] = sc_rt.fit_transform(
    -results_eval[["test_next_remaining_time_loss"]]
)
results_eval["nt_norm"] = sc_nt.fit_transform(
    -results_eval[["test_next_time_to_next_event_loss"]]
)

results_eval["mt_score"] = (
    results_eval["na_norm"] +
    results_eval["rt_norm"] +
    results_eval["nt_norm"]
)

# pick best multi-task run per (log × backbone) by mt_score
best_idx = (
    results_eval
    .groupby(["log", "backbone"])["mt_score"]
    .nlargest(1)
    .index.get_level_values(2)
)

best = results_eval.iloc[best_idx][cols].reset_index(drop=True)

os.makedirs("notebooks", exist_ok=True)
best_csv_path = os.path.join("notebooks", "best_models_multitask.csv")
best.to_csv(best_csv_path, index=False)

# %%
# multi-task models
METRICS = [
    'best_test_next_activity_acc',
    'best_test_next_remaining_time_loss',
    'best_test_next_time_to_next_event_loss'
]
def highlight_group_min(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for _, group in df.groupby("Dataset"):
        idxs = group.index
        for col in ["NA Acc.", "RT MSE", "NT MSE"]: #METRICS:
            min_val = group[col].max() if col in ["NA Acc.", "test_next_activity_acc", "best_test_next_activity_acc"] else group[col].min()
            styles.loc[idxs, col] = group[col].apply(
                lambda x: "font-weight: bold" if x == min_val else ""
            )
    return styles

import numpy as np
tmp = best.reset_index(drop=True).copy()
tmp = tmp[["log", "backbone", "fine_tuning"] + METRICS + ["total_params", "trainable_params"] + ["_runtime"]]

# seconds to hours
tmp["_runtime"] = (tmp["_runtime"] / 3600.0).round(3)

# params formatting
tmp["trainable_params"] = ((tmp.trainable_params / tmp.total_params) * 100).astype(int).astype(str) + "%"
tmp["total_params"] = tmp["total_params"].apply(lambda x: np.format_float_scientific(x, precision=1))
tmp["# params\n(\%trainable)"] = tmp["total_params"] + "(" + tmp["trainable_params"] + ")"

data = tmp.rename(columns={
    "log": "Dataset",
    "backbone": "Backbone",
    "best_test_next_activity_acc": "NA Acc.",
    "best_test_next_remaining_time_loss": "RT MSE",
    "best_test_next_time_to_next_event_loss": "NT MSE",
    "total_params": "# params\n(total)",
    "trainable_params": "% params\n(trainable)",   
    "_runtime": "Runtime (h)"
})

data.Dataset = data.Dataset.map({
    "BPI12": "BPI12",
    "BPI17": "BPI17",
    "BPI20PrepaidTravelCosts": "BPI20PTC",
    "BPI20RequestForPayment": "BPI20RfP",
    "BPI20TravelPermitData": "BPI20TPD",
})

data.Backbone = data.Backbone.map({
    "majority": "Majority",
    "rnn": "RNN",
    "transformer": "Transformer",
    "tabpfn": "TabPFN",
    "gpt2": "GPT2", 
    "gptneo-1b3": "GPT-Neo-1b3",
    "llama32-1b": "Llama3.2-1b",
    "qwen25-05b": "Qwen2.5-0.5b",
    "gemma-2-2b": "Gemma-2-2b",
})

data = data.sort_values(by=["Dataset", "Backbone"])

data.fine_tuning = data.fine_tuning.fillna("none")
data.fine_tuning = data.fine_tuning.map({
    "lora": "LoRA",
    "freeze": "Freezing",
    "none": "none"
})
data.Backbone = data.apply(lambda x: x["Backbone"] + " [" + x["fine_tuning"] + "]" if x["fine_tuning"] != "none" else x["Backbone"], axis=1)

data = data.drop(columns=["fine_tuning"]).reset_index(drop=True)
data = data.sort_values(by=["Dataset", "Backbone"])

os.makedirs("notebooks", exist_ok=True)
csv_path = os.path.join("notebooks", "big_table_v2.csv")
data.to_csv(csv_path, index=False)

# %%
import numpy as np

param_summary = (
    best.groupby("Dataset")[["total_params", "trainable_params"]]
    .first()
    .reset_index()
)

param_summary["trainable_percent"] = (
    (param_summary["trainable_params"] / param_summary["total_params"]) * 100
).astype(int).astype(str) + "%"

param_summary["# params\n(%trainable)"] = (
    param_summary["total_params_fmt"] + "(" + param_summary["trainable_percent"] + ")"
)

os.makedirs("notebooks", exist_ok=True)
csv_path = os.path.join("notebooks", "param_summary.csv")
param_summary.to_csv(csv_path, index=False)

print("=== PARAMETER SUMMARY PER DATASET ===")
print(param_summary.to_string(index=False))

# %%
# Loss curves
# multi-task models

loss_csv_path = os.path.join("notebooks", "final_loss_curves_multitask.csv")

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
    
    os.makedirs("notebooks", exist_ok=True)
    losses.to_csv(loss_csv_path, index=False)
    
# print a small sample into the log
print("=== MULTI-TASK LOSS CURVES (head) ===")
print(losses.head().to_string(index=False))

# %%
LOGS_TO_PLOT = ["BPI20PrepaidTravelCosts", "BPI12", "BPI17"]

HUE_MAP = {
    "majority":     "Majority",
    "rnn":          "RNN",
    "transformer":  "Transformer",
    "tabpfn":       "TabPFN",
    "gpt2":         "GPT2",
    "gptneo-1b3":   "GPT-Neo-1b3",
    "llama32-1b":   "Llama3.2-1b",
    "qwen25-05b":   "Qwen2.5-0.5b",
    "gemma-2-2b":   "Gemma-2-2b",
}

HUE_ORDER = [
    "Majority",
    "RNN",
    "Transformer",
    "TabPFN",
    "GPT2",
    "GPT-Neo-1b3",
    "Qwen2.5-0.5b",
    "Llama3.2-1b",
    "Gemma-2-2b",
]

l = losses.melt(
    id_vars=["log", "backbone", "epoch"],
    value_vars=["na_loss", "rt_loss", "nt_loss"],
    var_name="Loss",
    value_name="Value",
).copy()

l = l.dropna(subset=["Value"])
l["Backbone"] = l["backbone"].map(HUE_MAP)
l = l[l["Backbone"].notna()] 

LOSS_LABELS = {
    "na_loss": "NA Loss",
    "rt_loss": "RT Loss",
    "nt_loss": "NT Loss",
}

fig, axes = plt.subplots(3, len(LOGS_TO_PLOT), figsize=(12, 3), sharex=True)
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

        # remove legends in all but bottom-right subplot
        if (loss_name, log_name) != ("nt_loss", LOGS_TO_PLOT[-1]):
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            leg = ax.legend(title="", ncol=3, bbox_to_anchor=(1.05, -0.3))
            for line in leg.get_lines():
                line.set_linewidth(2.0)

plt.tight_layout()

# save to project folder
os.makedirs("notebooks/plots", exist_ok=True)
plot_path = os.path.join("notebooks", "plots", "loss_curves_multitask.png")
plt.savefig(plot_path, dpi=300)

# log to W&B
run = wandb.init(
    project=project,
    entity=entity,
    job_type="analysis",
    name="loss_curves_multitask",
)

wandb.log({"loss_curves_multitask": wandb.Image(fig)})
run.finish()