FROM busybox:latest

# Store code at /data/ (not /tmp/) so it survives volume mounts
# Code will be copied to /tmp/codeBase/ at runtime via docker-compose command
RUN mkdir -p /data/codeBase/llm \
    && mkdir -p /data/codeBase/embedding

# Copy LLM inference scripts
COPY inference.py /data/codeBase/llm/
COPY benchmark_icd10_bf16.py /data/codeBase/llm/

# Copy embedding inference scripts
COPY classify.py /data/codeBase/embedding/

# Copy README
COPY README.md /data/codeBase/
