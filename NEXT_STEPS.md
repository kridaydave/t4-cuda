# Tesla T4 GPU Physical Execution & Empirical Verification Protocol (H1 – H16)

This document specifies the concrete 5-stage execution plan for building, verifying, profiling, and benchmarking our complete suite of **custom Turing (`sm_75`) CUDA kernels (Hypotheses H1 through H16)** as soon as physical Tesla T4 GPU compute becomes available (e.g. Google Colab T4, AWS `g4dn.xlarge`, or RunPod T4 instance).

---

## 1. Fast Compile-Only Register & Assembly Audit (`nvcc`)

Before building PyTorch C++ extensions, execute static compilation to inspect PTX assembly register allocations and stack spill metrics.

```bash
# Static PTX compile audit for LOP3 unpacking, WMMA double buffering, and fused activation kernels
nvcc -arch=sm_75 -Xptxas -v -c research/src/t4_cuda_kernels.cu -o /tmp/t4_cuda_kernels.o
```

### Audit Target Metrics:
- **Registers per Thread**: Enforce target bound $\le 64$ registers/thread to preserve active warp occupancy.
- **Stack Spills**: Verify `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`. Any non-zero spill indicates register pressure or bad buffer indexing.
- **PTX Instruction Audit**: Verify `lop3.b32` opcodes emitted with constants `0x64046404` (INT3 LUT `0x6A`), `0x64086408` (INT4 signed LUT `0x6A`), and `0x64006400` (INT4 unsigned LUT `0xEA`). The FP8 E4M3 path is NO longer a `lop3.b32` op — it uses an integer `ADD` of `0x20002000` (+8 exponent re-bias) + bitwise `OR` with the sign word; audit for `IADD`/`OR` instead (SASS count unmeasured for the committed path).

---

## 2. PyTorch C++ Extension Build

Build the C++/CUDA extension locally using PyTorch's `setup.py` builder to verify C++ header linkages, PyBind11 signatures, CUDA stream types, and PyTorch ABI compatibility.

```bash
cd research/src && python3 setup.py build_ext --inplace
```

### Audit Target Metrics:
- Confirm `bindings.cpp` compiles cleanly without signature mismatches or missing symbol errors.
- Verify `t4_kernels.so` is created in `research/src/`.

---

## 3. On-GPU Differential Verification & Master Test Suite

Run differential verification comparing the custom LOP3 CUDA kernels directly against an independent PyTorch reference unpacker on the GPU across all 16 hypotheses (H1 to H16).

```bash
# Execute master experimental verification harness
python3 research/harness/master_experimental_verification.py
```

### KAT (Known Answer Test) Vectors & Tolerances:
- **INT4 KAT Vector (`0xA7C13E59`)**: Unpacks signed values `[-7, 5, -2, 3, 1, -4, 7, -6]` with 100% bit-exact accuracy.
- **INT3 KAT Vector (`0x64046404`)**: Unpacks 10 signed 3-bit values in $[-4, 3]$ with $0.0000$ error.
- **FP8 KAT Vector (`0xEA`)**: Re-biases FP8 `E4M3` exponent ($+8$) and shifts mantissa ($128 M_8$) across all 254 valid FP8 byte states with $0.0000$ float error.
- **Random Large Tensor Verification ($M=4096, N=4096$)**: Enforces `torch.allclose(kernel_out, ref_out, atol=1e-3)`.

---

## 4. Hardware Telemetry & Thermal/Clock Profiling (`nvidia-smi`)

Capture real SM boost clocks, TDP power draw, and active throttle reasons under tight execution loops to measure the physical hardware response of the T4 card.

```bash
# Launch background telemetry sampler (20ms sampling interval)
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv -lms 20 > /tmp/t4_telemetry.csv &
SAMPLER_PID=$!

# Run prefill & decode GEMM kernels in a tight execution loop for 10 seconds
python3 -c "
import torch, time
# Run master verification loop to stress GPU pipeline
import research.harness.master_experimental_verification as m
m.run_master_verification_suite()
"

kill $SAMPLER_PID
cat /tmp/t4_telemetry.csv | head -n 30
```

### Audit Target Metrics:
- **Power Draw**: Record real Watts consumed vs the 70W TDP ceiling (target: $\le 62.0\text{W}$ at 25% occupancy cap).
- **Observed SM Boost Clock**: Verify core clock remains locked at $1590\text{ MHz}$ without decaying down to $950\text{ MHz}$.
- **Throttle Reasons**: Verify `clocks_throttle_reasons.hw_power_brake = 0` and `sw_power_cap = 0`.

---

## 5. Nsight Compute (`ncu`) Roofline & Speed-of-Light Analysis

Once correctness and thermal behavior are measured, profile the kernels using Nsight Compute to obtain hardware-level Speed-of-Light (SOL) performance metrics.

```bash
ncu --metrics sm__throughput.of_peak_allocation_sol,dram__throughput.of_peak_allocation_sol,l1tex__data_bank_conflicts_pipe_lsu.sum,smsp__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained \
    python3 research/harness/master_experimental_verification.py
```

### Audit Target Metrics:
- **DRAM SOL %**: Verify GDDR6 memory bandwidth saturation reaches $\ge 94.8\%$ ($303.4\text{ GB/s}$ out of 320.0 GB/s peak).
- **SMEM Bank Conflicts**: Confirm `l1tex__data_bank_conflicts_pipe_lsu.sum = 0` (0 bank conflicts).
- **Fetch Warp Stalls**: Confirm fetch warp stall cycles $\le 14$ cycles/tile under Software Warp Specialization.
- **Tensor Core Pipe SOL %**: Verify FP16 Tensor Core active cycles reach $\ge 60.1\text{ TFLOPS}$ under FP8 emulation.
