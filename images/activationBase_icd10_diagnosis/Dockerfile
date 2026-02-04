FROM busybox:latest

# Store data at /data/ (not /tmp/) so it survives volume mounts
# Data will be copied to /tmp/activationBase/ at runtime via docker-compose command
RUN mkdir -p /data/activationBase

# Copy activation data
COPY activation_data.csv /data/activationBase/

# Copy README
COPY README.md /data/activationBase/
