#!/bin/bash
set -e

echo "[HOST] Triggering Docker build pipeline for SYNAPSE .deb..."

# Build the sterile docker image
docker build -t synapse_builder -f packaging/Dockerfile .

# Execute the container which acts as the build sandbox, 
# mounting the current directory so that the .deb outputs directly to the host
docker run --rm -v "$(pwd):/app" synapse_builder bash packaging/build_deb.sh

echo "[HOST] Docker task completed."
