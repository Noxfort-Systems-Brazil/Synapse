FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 and build dependencies
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    libgl1-mesa-dev \
    libglib2.0-0 \
    pkg-config \
    libcairo2-dev \
    fakeroot \
    dpkg-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
