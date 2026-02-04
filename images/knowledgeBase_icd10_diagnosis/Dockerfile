FROM busybox:latest

# Store data at /data/ (not /tmp/) so it survives volume mounts
# Data will be copied to /tmp/knowledgeBase/ at runtime via docker-compose command
RUN mkdir -p /data/knowledgeBase/llm_lora_adapter \
    && mkdir -p /data/knowledgeBase/embedding_model

# Copy LoRA adapter files (LLM)
# Note: These files are generated after fine-tuning and must be added before building
# Expected files: adapter_model.safetensors, adapter_config.json, tokenizer files
COPY llm_lora_adapter/ /data/knowledgeBase/llm_lora_adapter/

# Copy embedding model files
# Expected files: best_model.pt, label_encoder.pkl, encoder/ directory
COPY embedding_model/ /data/knowledgeBase/embedding_model/

# Copy README
COPY README.md /data/knowledgeBase/
