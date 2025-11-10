#!/bin/bash
#SBATCH --job-name=llm-peft-ppm_llama32-1b
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --mail-user=lennart.fertig@students.uni-mannheim.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --chdir=/ceph/lfertig/Thesis/notebook/llm-peft-ppm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Create log/cache dirs
mkdir -p logs .cache/huggingface .wandb

# Conda env
eval "$(/ceph/lfertig/miniconda3/bin/conda shell.bash hook)"
conda activate llm-peft-ppm

# Runtime
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_DIR="$PWD/.wandb"
export TOKENIZERS_PARALLELISM=false

# GPU Info
nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch,sys; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" || true

# Configuration
PARAMS_FILE="scripts/llama32-1b_params.txt" 
PY_MAIN="fertig_lennart_next_event_prediction.py"
PROJECT="llm-peft-ppm_llama32-1b"

grep -vE '^\s*#|^\s*$' "$PARAMS_FILE" | while IFS= read -r ARGS; do
  echo ">>> RUN: python $PY_MAIN $ARGS --project_name $PROJECT --wandb"
  python "$PY_MAIN" $ARGS --project_name "$PROJECT" --wandb
done