#!/bin/bash
#SBATCH --job-name=aibas_embedding_finetuning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40gb:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# == paths =================================================================
BASE_DIR="${SLURM_SUBMIT_DIR}"
SIF_IMAGE="${BASE_DIR}/container_embedding_finetuning.sif"
PROJECT_NAME="embedding_finetuning"

# == create dirs ===========================================================
mkdir -p ${BASE_DIR}/logs ${BASE_DIR}/wandb ${BASE_DIR}/embedding_in ${BASE_DIR}/embedding_out

# == offline env ===========================================================
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPUs available: $(nvidia-smi -L 2>/dev/null || echo 'checking inside container')"

# == run training with accelerate inside the container =====================
apptainer exec --nv \
    --bind "${BASE_DIR}:${BASE_DIR}" \
    "${SIF_IMAGE}" \
    python "${BASE_DIR}/embedding_finetuning.py" \
        --model_path "${BASE_DIR}/embedding_in" \
        --train_data "${BASE_DIR}/training_finetuning_embedding.json" \
        --val_data "${BASE_DIR}/validation_finetuning_embedding.json" \
        --output_dir "${BASE_DIR}/embedding_out" \
        --epochs 20 \
        --batch_size 32 \
        --lr 2e-5 \
        --max_length 512

echo "Job finished at $(date)"
