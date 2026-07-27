# Use a CUDA-enabled PyTorch runtime as the base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_NAME=facebook/sam3

# Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    sed \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install runpod
RUN pip install runpod

# Copy requirements.txt first to install python dependencies
COPY requirements.txt .

# Remove torch/torchvision/torchaudio from requirements.txt to preserve the pre-installed,
# CUDA-optimized PyTorch binaries that come with the base image, then install dependencies
RUN sed -i '/torch/d' requirements.txt && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Expose nothing (RunPod Serverless handles execution via polling)
CMD ["python", "-u", "handler.py"]
