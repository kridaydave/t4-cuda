#!/bin/bash
set -e

echo "=== Checking Environment ==="

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. No GPU?"
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. CUDA toolkit missing."
    exit 1
fi

if ! python -c "import torch" &> /dev/null; then
    echo "ERROR: PyTorch not found in Python environment."
    exit 1
fi

echo "Environment looks good."
echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
echo ""

echo "=== Building t4_kernels ==="
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRC_DIR="$SCRIPT_DIR/../src"

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: src/ directory not found at $SRC_DIR"
    exit 1
fi

cd "$SRC_DIR"
pip install -e .

echo ""
echo "=== Testing Build ==="
if python -c 'import t4_kernels; print("BUILD OK")'; then
    echo "Success: t4_kernels imported successfully."
else
    echo "ERROR: Failed to import t4_kernels after build."
    exit 1
fi
