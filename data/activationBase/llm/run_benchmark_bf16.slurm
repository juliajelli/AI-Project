#!/bin/bash
#SBATCH --job-name=bench_icd10_bf16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40gb:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ── paths ─────────────────────────────────────────────────────────────────
BASE_DIR="${SLURM_SUBMIT_DIR}"
SIF=${BASE_DIR}/container.sif

# ── create dirs ───────────────────────────────────────────────────────────
mkdir -p ${BASE_DIR}/logs ${BASE_DIR}/results_bf16

# ── offline env ───────────────────────────────────────────────────────────
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── run benchmark inside the container ────────────────────────────────────
apptainer exec --nv \
    --bind ${BASE_DIR}:${BASE_DIR} \
    ${SIF} \
    python ${BASE_DIR}/benchmark_icd10_bf16.py
