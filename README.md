# Extending Multi-Task Predictive Process Monitoring with Decoder-Only LLMs

## Overview

This repository contains code and scripts to fine-tune decoder-only large language models (LLMs) for multi-task predictive process monitoring (PPM) and to compare them against sequence baselines and tabular foundation models.

This thesis project is based on the original implementation by Oyamada et al. and extends it with an additional temporal prediction task, additional backbones, and additional baselines.

## Origin and Attribution

This project builds on the original repository and approach:

- Original GitHub repository: https://github.com/raseidi/llm-peft-ppm
- Original paper (arXiv): https://doi.org/10.48550/arXiv.2509.03161

### Thesis extensions (relative to the original work)

Compared to the original implementation, this thesis project:

1. Extends the multi-task setting by adding next event time alongside next activity and remaining tim, resulting in three prediction heads.
2. Evaluates additional decoder-only LLM backbones under the same interface, including zero-shot and few-shot adaptation settings.
3. Includes tabular foundation models as additional baselines.

## System and Tooling

- Tested on **Ubuntu 24.04**
- Python **3.12**
- Local environment management via **uv**: https://docs.astral.sh/uv/guides/install-python/

## Installation

You can set up the project using either `requirements.txt` (pip/uv) or the provided conda environment files.

### Option A: `requirements.txt` (pip/uv)

Create and activate a virtual environment (example with `uv`):

```bash
uv venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

### Option B: Conda environments (recommended for reproducibility)

This repository provides two conda environment files:

- `env-llm-peft-ppm.yml`
- `env-llm-peft-ppm-saprpt.yml`

Create and activate one of them:

```bash
conda env create -f env-llm-peft-ppm.yml
conda activate env-llm-peft-ppm
```

or

```bash
conda env create -f env-llm-peft-ppm-saprpt.yml
conda activate env-llm-peft-ppm-saprpt
```

Notes:

- Use the conda environment files for thesis-grade reproducibility.
- Use `requirements.txt` for lightweight/local debugging setups.

## Repository Structure

```text
.
├── data/                                       # Event logs (automatically downloaded)
├── scripts/                                    # Experiment scripts and configs
│   ├── *.sh
│   └── *.txt
├── notebooks/                                  # Analysis notebooks
├── ppm/                                        # Source code
├── fertig_lennart_next_event_prediction.py     # Main training script
├── requirements.txt                            # Python dependencies (pip/uv)
├── env-llm-peft-ppm.yml                        # Conda environment (reproducibility)
├── env-llm-peft-ppm-saprpt.yml                 # Conda environment variant (reproducibility)
└── README.md                                   # This file
```

## Data

Five public event logs are used. They are downloaded via `skpm` into `data/<LOG>/`:

SkPM documentation: https://skpm.readthedocs.io/en/latest/examples/01_data_api.html

Event logs:

- BPI20PTC (Prepaid Travel Costs): https://doi.org/10.4121/uuid:5d2fe5e1-f91f-4a3b-ad9b-9e4126870165
- BPI20RfP (Request for Payment): https://doi.org/10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030
- BPI20TPD (Permit Data): https://doi.org/10.4121/uuid:ea03d361-a7cd-4f5e-83d8-5fbdf0362550
- BPI12: https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f
- BPI17: https://doi.org/10.4121/uuid:c2c3b154-ab26-4b31-a0e8-8f2350ddac11

## Usage

### RNN baseline

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
  --continuous_targets remaining_time next_event_time
```

### LLM fine-tuning (LoRA)

To use Hugging Face-hosted model weights, a Hugging Face token is required:

Hugging Face tokens: https://huggingface.co/docs/hub/en/security-tokens

Supported ways to provide it:

1. Create an `.env` file in the repository root:

```text
HF_TOKEN=<YOUR_TOKEN>
```

2. Export an environment variable:

```bash
export HF_TOKEN="<YOUR_TOKEN>"
```

3. Use environment-variable access in code (recommended pattern already used in the project):

`HF_TOKEN = os.getenv("HF_TOKEN")` (see `ppm/models/models.py`)

Minimal local debugging configuration (reduce `batch_size` if GPU memory is limited):

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
  --continuous_targets remaining_time next_event_time \
  --fine_tuning lora \
  --r 2 \
  --lora_alpha 4
```

Weights & Biases logging can be enabled with `--wandb`.

## Hyperparameter Search and Reproducibility

Experiment scripts and configuration grids are located in:

- `scripts/*.sh`
- `scripts/*.txt`

They document how runs were executed and can be used to reproduce experiments locally or on compute infrastructure.

## Results

Metrics, exports, and analysis notebooks are stored under:

- `results/`

Plots and analysis are available in:

- `results/results.ipynb`

## Citation

If you use or build upon this repository, cite the original work:

- Original GitHub: https://github.com/raseidi/llm-peft-ppm
- Original paper: https://doi.org/10.48550/arXiv.2509.03161

## Contact

For questions or feedback or open an issue in this repository.