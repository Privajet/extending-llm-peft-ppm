#!/bin/bash
#SBATCH --job-name=BPI12_llm-peft-ppm_qwen3-4b_training
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

# Helpful runtime settings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_DIR="$PWD/.wandb"
export TOKENIZERS_PARALLELISM=false

# GPU Info (optional)
nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch,sys; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" || true

PROJECT="BPI12_llm-peft-ppm_qwen3-4b"
LR=(0.00005)
BATCH_SIZE=8
EPOCHS=10
# R=(256)
R=(8)
FINE_TUNING=("lora" "freeze")
declare -a FREEZE_LAYERS=(
    "-1 -2"
    "0 1"
    "0"
    "-1"
    null
)

# declare -a DATASETS=(BPI12 BPI17 BPI20PrepaidTravelCosts BPI20TravelPermitData BPI20RequestForPayment)
declare -a DATASETS=(BPI12)

# python fetch_wandb.py --project $PROJECT

for dataset in "${DATASETS[@]}"
do
    for lr in "${LR[@]}"
    do
        for fine_tuning in "${FINE_TUNING[@]}"
        do
            cmd="--dataset $dataset \
                --backbone qwen3-4b \
                --embedding_size 896 \
                --hidden_size 896 \
                --categorical_features activity \
                --categorical_targets activity \
                --continuous_features all \
                --continuous_targets remaining_time time_to_next_event \
                --strategy sum \
                --lr $lr \
                --batch_size $BATCH_SIZE \
                --epochs $EPOCHS \
                --fine_tuning $fine_tuning \
                --project_name $PROJECT"

            if [[ $fine_tuning == "lora" ]]; then
                for r in "${R[@]}"
                do
                    cmd2="$cmd \
                    --r $r \
                    --lora_alpha $(( r*2 ))"
                    python fertig_lennart_next_event_prediction.py $cmd2 --wandb
                    echo $cmd2 --wandb
                done
            else
                for freeze_layers in "${FREEZE_LAYERS[@]}"
                do
                    if [ "$freeze_layers" == "null" ]; then
                        cmd2=$cmd
                    else
                        cmd2="$cmd --freeze_layers $freeze_layers"
                    fi
                    echo $cmd2 --wandb
                    python fertig_lennart_next_event_prediction.py $cmd2 --wandb
                done
            fi
        done
    done
done