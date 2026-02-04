FROM busybox:latest

# Store data at /data/ (not /tmp/) so it survives volume mounts
# Data will be copied to /tmp/learningBase/ at runtime via docker-compose command
RUN mkdir -p /data/learningBase/train/llm \
    && mkdir -p /data/learningBase/train/embedding \
    && mkdir -p /data/learningBase/validation/llm \
    && mkdir -p /data/learningBase/validation/embedding

# Copy training data
COPY training_finetuning_llm.jsonl /data/learningBase/train/llm/
COPY training_finetuning_embedding.json /data/learningBase/train/embedding/

# Copy validation/test data
COPY validation_finetuning_llm.jsonl /data/learningBase/validation/llm/
COPY validation_finetuning_embedding.json /data/learningBase/validation/embedding/

# Copy README
COPY README.md /data/learningBase/
