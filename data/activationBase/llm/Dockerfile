FROM nvcr.io/nvidia/pytorch:24.07-py3

# Arbeitsverzeichnis
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

# Environment variables
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    WANDB_MODE=offline

# Copy inference & benchmark scripts
COPY inference.py /workspace/inference.py
COPY benchmark_icd10_bf16.py /workspace/benchmark_icd10_bf16.py

# Default execution
CMD ["python", "inference.py"]
