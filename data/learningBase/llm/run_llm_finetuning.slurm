#!/bin/bash
#SBATCH --job-name=aibas_llm_finetuning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40gb:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# == paths =================================================================
BASE_DIR="${SLURM_SUBMIT_DIR}"
SIF_IMAGE=${BASE_DIR}/container_llm_finetuning.sif
TRAIN_SCRIPT=${BASE_DIR}/llm_finetuning.py
LLM_PATH=${BASE_DIR}/llm_model
PROJECT_NAME="llm_finetuning"

# == create dirs ===========================================================
mkdir -p ${BASE_DIR}/logs ${BASE_DIR}/wandb ${BASE_DIR}/llm_model_in ${BASE_DIR}/llm_model_out

# == offline env ===========================================================
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPUs available: $(nvidia-smi -L 2>/dev/null || echo 'checking inside container')"

# == run training with accelerate inside the container =====================
apptainer exec --nv \
    --bind ${BASE_DIR}:${BASE_DIR} \
    ${SIF_IMAGE} \
    accelerate launch \
        --multi_gpu \
        --num_processes 3 \
        ${TRAIN_SCRIPT}

echo "Job finished at $(date)"
