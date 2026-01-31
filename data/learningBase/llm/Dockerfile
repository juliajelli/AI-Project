FROM nvcr.io/nvidia/pytorch:24.07-py3

# Arbeitsverzeichnis im Container
WORKDIR /workspace

# Python Dependencies
RUN pip install --no-cache-dir \
    transformers==4.57.6 \
    datasets==3.2.0 \
    accelerate==1.7.0 \
    peft==0.14.0 \
    trl==0.13.0 \
    bitsandbytes==0.45.0 \
    wandb==0.19.0 \
    mistral-common \
    scipy

# Environment variables (offline / reproducibility)
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    WANDB_MODE=offline

# Copy fine-tuning script into container
COPY llm_finetuning.py /workspace/llm_finetuning.py

# Container start command
CMD ["python", "llm_finetuning.py"]
