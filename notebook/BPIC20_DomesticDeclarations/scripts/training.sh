#!/bin/bash
#SBATCH --job-name=BPIC20_DomesticDeclarations_NT_Transformer_training
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --mail-user=lennart.fertig@students.uni-mannheim.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --chdir=/ceph/lfertig/Thesis
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Caches & Logging (workdir)
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_DIR="$PWD/.wandb"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$WANDB_DIR" logs

# Conda initialisieren und Env aktivieren
eval "$(/ceph/lfertig/miniconda3/bin/conda shell.bash hook)"

# >>> HIER die gewünschte Env wählen <<<
ENV_NAME=${ENV_NAME:-thesis-baselines}
# ENV_NAME=${ENV_NAME:-thesis-llm}
# ENV_NAME=${ENV_NAME:-thesis-llm-qwen}
conda activate "$ENV_NAME"

# Threads
export PYTHONPATH="/ceph/lfertig/Thesis:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# Faster HF downloads (if available on your cluster)
export HF_HUB_ENABLE_HF_TRANSFER=1
# Better CUDA memory behavior on PyTorch 2.x
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU Info (optional)
nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch,sys; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" || true

# BASELINES
# Majority:
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_majority_class_ACT_BPIC20_DomesticDeclarations.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_majority_class_NT_BPIC20_DomesticDeclarations.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_majority_class_RT_BPIC20_DomesticDeclarations.py

# NGram:
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_ngram_ACT_BPIC20_DomesticDeclarations.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_ngram_NT_BPIC20_DomesticDeclarations.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_ngram_RT_BPIC20_DomesticDeclarations.py

# LSTM:
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_lstm_predict_ACT_BPIC20_DomesticDeclarations.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_lstm_predict_NT_BPIC20_DomesticDeclarations.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_lstm_predict_RT_BPIC20_DomesticDeclarations.py

# ProcessTransformer:
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_processTransformer_ACT_BPIC20_DomesticDeclarations.py
srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_processTransformer_NT_BPIC20_DomesticDeclarations.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_processTransformer_RT_BPIC20_DomesticDeclarations.py

# TabPFN:
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_TabPFN_ACT_BPIC20_DomesticDeclarations.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_TabPFN_NT_BPIC20_DomesticDeclarations.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/baseline/fertig_lennart_baseline_TabPFN_RT_BPIC20_DomesticDeclarations.py

# LLM (Zero-Shot / Few-Shot / Fine-Tuning) nur mit thesis-llm:
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/fertig_lennart_gpt-neo-1.3B_ACT_Zero-Shot-Learning_BPIC20_DomesticDeclarations.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/fertig_lennart_gpt-neo-1.3B_ACT_Few-Shot-Learning_BPIC20_DomesticDeclarations.py
# srun python -u /ceph/lfertig/Thesis/notebook/BPIC20_DomesticDeclarations/notebook/fertig_lennart_gpt-neo-1.3B_ACT_Fine-Tuning_SFT_Trainier_BPIC20_DomesticDeclarations.py