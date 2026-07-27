#!/bin/bash

# Force fail on error
set -e

echo "=========================================================="
echo "Initializing SAM3 RunPod Serverless Worker"
echo "=========================================================="

# Display System and GPU details
echo "--- GPU Details ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi utility not found. Checking PyTorch CUDA status instead."
fi

echo "--- Python/PyTorch Diagnostic ---"
python -u -c "
import torch
print('Python version:', torch.sys.version.split()[0])
print('PyTorch version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Device Name:', torch.cuda.get_device_name(0))
    print('CUDA Device Count:', torch.cuda.device_count())
"

# Set default model name if not provided
export MODEL_NAME=${MODEL_NAME:-"facebook/sam3"}
echo "--- Worker configuration ---"
echo "Target Model Config: $MODEL_NAME"

# Run the s handler
echo "Starting handler.py..."
exec python -u handler.py
