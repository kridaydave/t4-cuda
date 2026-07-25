# Tesla T4 GPU Execution & Empirical Verification Protocol

This document details the concrete 5-stage execution plan for building, verifying, profiling, and benchmarking our custom **Turing (`sm_75`) CUDA kernels** on a physical Tesla T4 GPU (e.g. Google Colab T4 instance or AWS `g4dn`).

---

## 1. Fast Compile-Only Register & Assembly Audit (`nvcc`)

Before building Python extensions, perform a fast static compilation check to inspect ptxas register allocation and stack spill metrics.

```bash
nvcc -arch=sm_75 -Xptxas -v -c research/src/kernels/lop3_dequant.cu -o /tmp/lop3_dequant.o
```

### Audit Targets:
- **Actual Register Count per Thread**: Check register allocation reported by `ptxas` against theoretical target bounds ($\le 64$ registers/thread).
- **Stack Spills**: Verify `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`. Any non-zero spill indicates register pressure or bad buffer indexing.

---

## 2. PyTorch C++ Extension Build

Build the C++/CUDA extension locally using PyTorch's `setup.py` builder to verify C++ header linkages, PyBind11 signatures, CUDA stream types, and PyTorch ABI compatibility.

```bash
cd research/src && python3 setup.py build_ext --inplace
```

### Audit Targets:
- Confirm `bindings.cpp` compiles cleanly without signature mismatches or missing symbol errors.
- Verify `t4_kernels.so` is created in `research/src/`.

---

## 3. On-GPU Differential Verification (Kernel vs Independent Reference)

Run differential verification comparing the custom LOP3 CUDA kernel directly against an independent PyTorch reference unpacker on the GPU.

```python
import torch
import t4_kernels
from research.harness.verify_dequant import dequantize_u4_reference, dequantize_s4_reference

# 1. Run KAT Vectors First (Hex vectors: 0xA7C13E59, 0xF817E29A)
w_test = torch.tensor([0xA7C13E59, 0xF817E29A], dtype=torch.int32, device="cuda")
scale_test = torch.tensor([0.25, 0.5], dtype=torch.float16, device="cuda")
zero_test  = torch.tensor([2.0, 0.0], dtype=torch.float16, device="cuda")

kernel_out = t4_kernels.dequantize_u4(w_test[:1], scale_test[:1], zero_test[:1])
ref_out = torch.tensor(dequantize_u4_reference([0xA7C13E59], [0.25], [2.0]), dtype=torch.float16, device="cuda")

assert torch.allclose(kernel_out, ref_out, atol=1e-3), "Unsigned KAT Mismatch on GPU!"
```

### Audit Targets:
- KAT hex vector match (`0xA7C13E59`, `0xF817E29A`) guarantees that register layouts and `half2` memory packing match theoretical expectations.
- Large-tensor random verification (`M=4096, N=4096`) enforces `torch.allclose(kernel_out, ref_out, atol=1e-3)`.

---

## 4. Hardware Telemetry & Thermal/Clock Profiling (`nvidia-smi`)

Capture real SM boost clocks, TDP power draw, and active throttle reasons under tight execution loops to measure the physical hardware response of the T4 card.

```bash
# Launch background telemetry sampler (20ms interval)
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv -lms 20 > /tmp/t4_telemetry.csv &
SAMPLER_PID=$!

# Run kernel in a tight execution loop for 5 seconds
python3 -c "
import torch, t4_kernels, time
w = torch.randint(0, 0x7FFFFFFF, (65536,), dtype=torch.int32, device='cuda')
s = torch.rand((65536,), dtype=torch.float16, device='cuda')
z = torch.zeros((65536,), dtype=torch.float16, device='cuda')
start = time.time()
while time.time() - start < 5.0:
    _ = t4_kernels.dequantize_u4(w, s, z)
torch.cuda.synchronize()
"

kill $SAMPLER_PID
cat /tmp/t4_telemetry.csv | head -n 30
```

### Audit Targets:
- **Observed SM Clock**: Record actual SM boost clocks under high occupancy vs capped occupancy.
- **Power Draw**: Record real Watts consumed vs the 70W TDP ceiling.
- **Throttle Reasons**: Check whether `HW Power Brake` or `Sw Power Cap` active throttle flags engage.

---

## 5. Nsight Compute (`ncu`) Roofline & Speed-of-Light Analysis

Once correctness and thermal behavior are measured, profile the kernel using Nsight Compute to obtain hardware-level Speed-of-Light (SOL) performance metrics.

```bash
ncu --metrics sm__throughput.of_peak_allocation_sol,dram__throughput.of_peak_allocation_sol,smsp__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained \
    python3 research/harness/verify_dequant.py
```

### Audit Targets:
- **DRAM SOL %**: Measure effective GDDR6 memory bandwidth saturation against the 320 GB/s peak.
- **Tensor Core Pipe SOL %**: Measure HMMA pipe active cycles.
- **Occupancy & Scheduler Stat**: Verify active warps per SM and warp stall causes.
