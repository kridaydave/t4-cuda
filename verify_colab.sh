#!/usr/bin/env bash
# Tesla T4 CUDA Research & Kernel Suite: Complete Colab Verification Pipeline
set -e

echo "=========================================================================="
echo "  Tesla T4 CUDA Research & Kernel Suite: Complete Verification Pipeline"
echo "=========================================================================="

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:$REPO_DIR/src:$PYTHONPATH"

echo ""
echo "--> [1/5] Running Bit-Exact IEEE-754 Math Proofs & KAT Harness..."
python3 harness/verify_dequant.py

echo ""
echo "--> [2/5] Running Microarchitectural Roofline Simulation..."
python3 src/t4_roofline_and_kernel_benchmarks.py

echo ""
echo "--> [3/5] Compiling & Running Standalone CUDA Micro-benchmarks (sm_75)..."
nvcc -O3 -arch=sm_75 src/t4_microbenchmarks.cu -o t4_microbenchmark
./t4_microbenchmark

echo ""
echo "--> [4/5] Building & Installing PyTorch CUDA Extension (t4_kernels)..."
cd "$REPO_DIR/src"
python3 setup.py build_ext --inplace
cp -f *.so "$REPO_DIR/" 2>/dev/null || true
cp -f *.so "$REPO_DIR/src/" 2>/dev/null || true
python3 setup.py develop --user >/dev/null 2>&1 || true
cd "$REPO_DIR"

echo ""
echo "--> [5/5] Running On-GPU Differential Verification & Stress Test..."
python3 -c "
import sys, os
sys.path.insert(0, '$REPO_DIR/src')
sys.path.insert(0, '$REPO_DIR')

import torch
import t4_kernels
from harness.verify_dequant import dequantize_u4_reference, dequantize_s4_reference

print('[PyTorch GPU Check] Device Name:', torch.cuda.get_device_name(0))

# 1. Known Answer Test (KAT) Hex Vectors on GPU (convert uint64 hex to int32 bit pattern)
w_test = torch.tensor([0xA7C13E59, 0xF817E29A], dtype=torch.int64).to(torch.int32).to('cuda')
scale_test = torch.tensor([0.25, 0.5], dtype=torch.float16, device='cuda')
zero_test  = torch.tensor([2.0, 0.0], dtype=torch.float16, device='cuda')

kernel_out = t4_kernels.dequantize_u4(w_test[:1], scale_test[:1], zero_test[:1])
ref_out = torch.tensor(dequantize_u4_reference([0xA7C13E59], [0.25], [2.0]), dtype=torch.float16, device='cuda')

print('Kernel Output:   ', kernel_out)
print('Reference Output:', ref_out)
assert torch.allclose(kernel_out, ref_out, atol=1e-3), 'Unsigned KAT Mismatch on GPU!'
print('>> [GPU SUCCESS] On-GPU KAT Match Verified!')

# 2. Large Tensor Stress Test (4096 x 4096)
w_large = torch.randint(0, 0x7FFFFFFF, (4096 * 512,), dtype=torch.int32, device='cuda')
s_large = torch.rand((4096 * 512,), dtype=torch.float16, device='cuda')
z_large = torch.zeros((4096 * 512,), dtype=torch.float16, device='cuda')

out_gpu = t4_kernels.dequantize_u4(w_large, s_large, z_large)
torch.cuda.synchronize()
print('>> [GPU STRESS TEST] Completed 4096x4096 dequantization without crashes. Shape:', out_gpu.shape)
"

echo ""
echo "=========================================================================="
echo "  [ALL VERIFICATIONS PASSED SUCCESSFULLY]"
echo "=========================================================================="
