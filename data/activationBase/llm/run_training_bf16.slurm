#!/bin/bash
#SBATCH --job-name=aibas_llm_ft_bf16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40gb:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ── paths ─────────────────────────────────────────────────────────────────
BASE_DIR="${SLURM_SUBMIT_DIR}"
SIF=${BASE_DIR}/container.sif
TRAIN_SCRIPT=${BASE_DIR}/train_bf16.py

# ── create dirs ───────────────────────────────────────────────────────────
mkdir -p ${BASE_DIR}/logs ${BASE_DIR}/output_bf16 ${BASE_DIR}/wandb

# ── offline env ───────────────────────────────────────────────────────────
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# ── run training with accelerate inside the container ─────────────────────
apptainer exec --nv \
    --bind ${BASE_DIR}:${BASE_DIR} \
    ${SIF} \
    accelerate launch \
        --multi_gpu \
        --num_processes 3 \
        ${TRAIN_SCRIPT}
