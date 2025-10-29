#!/bin/bash
#SBATCH --job-name=BPI_Challenge_2012C_ACT_LSTM_training
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --mail-user=lennart.fertig@students.uni-mannheim.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --chdir=/ceph/lfertig/Thesis
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# caches & logging
export HF_HOME="$SLURM_SUBMIT_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR="$SLURM_SUBMIT_DIR/.wandb"

set -euo pipefail

# Always run from the directory you submitted 'sbatch' in:
cd "$SLURM_SUBMIT_DIR"

# Ensure log/cache dirs exist
mkdir -p logs .wandb .cache/huggingface

# Activate your conda env (adjust conda path if needed)
eval "$(/ceph/lfertig/miniconda3/bin/conda shell.bash hook)"
ENV_NAME=${ENV_NAME:-thesis-baselines}
# ENV_NAME=${ENV_NAME:-thesis-llm}
conda activate "$ENV_NAME"

# Nice-to-have threading/env
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export WANDB_DIR="$SLURM_SUBMIT_DIR/.wandb"
export HF_HOME="$SLURM_SUBMIT_DIR/.cache/huggingface"

# GPU info (optional)
nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


# BASELINES
# Majority:
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_majority_class_ACT_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_majority_class_NT_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_majority_class_RT_BPI_Challenge_2012C.py

# NGram:
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_ngram_ACT_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_ngram_NT_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_ngram_RT_BPI_Challenge_2012C.py

# LSTM:
srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_lstm_predict_ACT_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_lstm_predict_NT_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_lstm_predict_RT_BPI_Challenge_2012C.py

# ProcessTransformer:
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_processTransformer_ACT_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_processTransformer_NT_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_processTransformer_RT_BPI_Challenge_2012C.py

# TabPFN:
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_TabPFN_ACT_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_TabPFN_NT_BPI_Challenge_2012C.py
# srun python - u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/baseline/fertig_lennart_baseline_TabPFN_RT_BPI_Challenge_2012C.py

# LLM (Zero-Shot / Few-Shot / Fine-Tuning) nur mit thesis-llm:
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/gpt-neo-1.3B/fertig_lennart_gpt-neo-1.3B_ACT_Zero-Shot-Learning_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/gpt-neo-1.3B/fertig_lennart_gpt-neo-1.3B_NT_Zero-Shot-Learning_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/gpt-neo-1.3B/fertig_lennart_gpt-neo-1.3B_RT_Zero-Shot-Learning_BPI_Challenge_2012C.py

# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/gpt-neo-1.3B/fertig_lennart_gpt-neo-1.3B_ACT_Few-Shot-Learning_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/gpt-neo-1.3B/fertig_lennart_gpt-neo-1.3B_NT_Few-Shot-Learning_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/gpt-neo-1.3B/fertig_lennart_gpt-neo-1.3B_RT_Few-Shot-Learning_BPI_Challenge_2012C.py

# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/gpt-neo-1.3B/fertig_lennart_gpt-neo-1.3B_ACT_Fine-Tuning_SFT_Trainier_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/gpt-neo-1.3B/fertig_lennart_gpt-neo-1.3B_NT_Fine-Tuning_SFT_Trainier_BPI_Challenge_2012C.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPI_Challenge_2012C/notebook/gpt-neo-1.3B/fertig_lennart_gpt-neo-1.3B_RT_Fine-Tuning_SFT_Trainier_BPI_Challenge_2012C.py