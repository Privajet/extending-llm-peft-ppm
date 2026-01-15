# Fine-Tuning LLMs for Multi-Task Predictive Process Monitoring

## Overview

* This repo has code and scripts to fine-tune large language models (LLMs) for multi-task PPM.
* We use [uv](https://docs.astral.sh/uv/guides/install-python/) to manage our local environment.
* Tested only on Ubuntu 24.04 using Python 3.12.

## Requirements

Install all dependencies provided in requirements.txt:

```bash
# Base Python version
python==3.12

# Core deps
numpy==1.26
pandas==2.2
scikit-learn==1.5
matplotlib
seaborn
ipykernel

# LLM / Deep learning stack
peft==0.14.0

# Torch
torch>=2.7.0

# Transformers
transformers>=4.52.4

wandb==0.19.8
pydantic==2.11.5
torchmetrics==1.7.4
datasets==4.0.0

# Git-based dependencies
skpm @ git+https://github.com/raseidi/skpm.git
tabpfn
tabpfn-extensions @ git+https://github.com/PriorLabs/tabpfn-extensions.git
sap_rpt_oss @ git+https://github.com/SAP-samples/sap-rpt-1-oss.git
```

## Scripts and Structure

```
.
├── data/                                       # Event logs (automatically downloaded)
├── scripts/                                    # Experiment scripts and configs
│   ├── *.sh                        
│   └── *.txt                                         
├── notebooks/                                  # Analysis notebooks
├── ppm/                                        # Source code
├── fertig_lennart_next_event_prediction.py     # Main training script
├── next_event_prediction.py                    # Original training script
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

## Data

We use five public event logs. They will be downloaded via [SkPM](https://skpm.readthedocs.io/en/latest/examples/01_data_api.html) under `data/<LOG>/`:

* [BPI20PTC](https://doi.org/10.4121/uuid:5d2fe5e1-f91f-4a3b-ad9b-9e4126870165) (Prepaid Travel Costs)
* [BPI20RfP](https://doi.org/10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030) (Request for Payment)
* [BPI20TPD](https://doi.org/10.4121/uuid:ea03d361-a7cd-4f5e-83d8-5fbdf0362550) (Permit Data)
* [BPI12](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f)
* [BPI17](https://doi.org/10.4121/uuid:c2c3b154-ab26-4b31-a0e8-8f2350ddac11)

## Usage

**RNN baseline**

```bash
python fertig_lennart_next_event_prediction.py \
  --dataset BPI20PrepaidTravelCosts \
  --backbone rnn \
  --embedding_size 32 \
  --hidden_size 128 \
  --lr 0.0005 \
  --batch_size 64 \
  --epochs 25 \
  --categorical_features activity \
  --continuous_features all \
  --categorical_targets activity \
  --continuous_targets remaining_time
```

**LLM fine-tuning**

In order to use LLMs, you need a [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens). A few options on how to use it:

* Create an `.env` file in the root of this repository and write your token like `HF_TOKEN=<YOUR_TOKEN>`
* Export a local variable `export HF_TOKEN="<YOUR_TOKEN>"`
* Hard code it [here](notebook/llm-peft-ppm/ppm/models/models.py) under "HF_TOKEN = os.getenv("HF_TOKEN")"

For local debugging purposes, try the tiny setup below with a small `r` value for `BPI20PrepaidTravelCosts` and `qwen25-05b`. If it doesn't fit your GPU memory, keep decreasing the `batch_size` (=4 uses less than 2gb). 

```bash
python fertig_lennart_next_event_prediction.py \
  --dataset BPI20PrepaidTravelCosts \
  --backbone qwen25-05b \
  --embedding_size 896 \
  --hidden_size 896 \
  --lr 0.00005 \
  --batch_size 64 \
  --epochs 1 \
  --categorical_features activity \
  --continuous_features all \
  --categorical_targets activity \
  --continuous_targets remaining_time \
  --fine_tuning lora \
  --r 2 \
  --lora_alpha 4
```

Alternatively, use the argument `--wandb` to enable wandb.

### Hyperparameter search

Check `scripts/*.sh` and `scripts/*.txt` to see how to reproduce jobs or run other configurations locally.

## Results

All metrics and analysis notebooks are in the `results/` folder. Check [this notebook](results/results.ipynb) for plots.

## Contact

For questions or feedback, reach me at [lennart.fertig@students.uni-mannheim.de](mailto:lennart.fertig@students.uni-mannheim.de) or open an issue here.

Orginal Link und GitHub hier rein packen. Paper Cite 
