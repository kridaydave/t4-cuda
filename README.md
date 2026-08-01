# `t4-cuda`: Turing CC 7.5 Extreme CUDA & PTX Kernel Optimizations

[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20Tesla%20T4%20(TU104)-76B900.svg)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![Compute Capability](https://img.shields.io/badge/Compute%20Capability-7.5-blue.svg)](https://developer.nvidia.com/cuda-gpus)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An extreme, microarchitecturally-optimized CUDA C++ and PTX assembly kernel suite custom-tailored for **NVIDIA Tesla T4 GPUs** (Turing CC 7.5, TU104 die, 40 SMs, 320 Tensor Cores, 70W TDP).

> **Research state**: v9.2.0 (2026-08-01). Systems hypothesis register H1–H17 / H20 / H22–H30 with novelty re-audit; persona hypotheses H18/H19/H21/H26/H31–H33 split to [`persona-hypotheses.yaml`](file:///home/kriday/Desktop/epoch-1/research/persona-hypotheses.yaml). See [research/findings.md](file:///home/kriday/Desktop/epoch-1/research/findings.md) §7 and [research/literature/survey_2026_07_29_novelty_reaudit_and_new_gaps.md](file:///home/kriday/Desktop/epoch-1/research/literature/survey_2026_07_29_novelty_reaudit_and_new_gaps.md).

---

## Executive Summary & Architectural Motivation

Millions of Tesla T4 GPUs are active in cloud infrastructure (Google Colab, AWS `g4dn`, GCP, Azure). However, modern LLM inference engines (e.g., vLLM, Marlin, FlashInfer, CUTLASS 3.x) hardcode Ampere/Hopper hardware primitives (`cp.async`, $K=16$ matrix shapes) that **do not physically exist on Turing CC 7.5**. When executed on T4, modern frameworks either crash or drop back to un-optimized PyTorch fallbacks.

Furthermore, on passively cooled 70W T4 GPUs, standard CUDA kernels launching at 100% thread occupancy trigger hardware power throttling (**NVIDIA Power Management / NVPM**), dropping core clocks from **1590 MHz down to ~950 MHz** (a 38% loss in TFLOPS).

**`t4-cuda`** addresses these hardware constraints through custom PTX assembly, 70W power-aware regime-split occupancy caps, fused backward GEMM optimizer kernels, and T4-specific pipeline integrations.

---

## Analytical Assessment & Prior Art Scorecard

> **Note on Research Rigor**: Performance metrics below represent theoretical modeling predictions and mathematical derivations. Empirical verification on physical T4 GPUs is tracked in [NEXT_STEPS.md](file:///home/kriday/Desktop/epoch-1/research/NEXT_STEPS.md).

| Technique / Engineering Component | Microarchitectural Mechanism | Target Hardware Gain | Status & Prior Art Attribution |
| :--- | :--- | :--- | :--- |
| **Power-Aware Occupancy Cap & Regime Split** | Capping occupancy at **25.0%** for compute-bound prefill ($M\ge 2048$), relaxing to **50%–75%** for memory-bound decode ($M=1$). | **Locks 1590 MHz boost clock** (prevents throttling to ~950 MHz); predicted **$1.47\times$ prefill speedup**. | **Original Contribution**: Novel power-pacing strategy for passively cooled 70W T4 GPUs. *(Status: Analytically Modeled)* |
| **Fused Backward GEMM + AdamW Kernel** | Accumulates $\nabla W$ in register fragments across $K$ loop; updates AdamW states in registers. | **21.4% DRAM bandwidth saving** (28 B/param down to 22 B/param). | **Original Contribution**: Fused register-level backward GEMM + optimizer scheme. *(Status: Analytically Modeled)* |
| **Signed INT4 Dequant (`LOP3` LUT `0x6A`)** | Single-cycle sign bit 3 inversion + FP16 magic exponent `0x64086408` insertion. | **1 SASS cycle** ($2.5\times$ instruction reduction over `bfe`). | **T4 Adaptation**: Extends established FP16 magic number insertion to signed INT4. *(Status: Mathematically Verified)* |
| **Unsigned INT4 Magic Exponent (`0x64006400`)** | Direct mantissa injection bypassing integer-to-float conversion pipe. | **2.5$\times$ instruction reduction** (20 down to 8 SASS insts). | **Established Prior Art**: Standard technique in ExLlama / Marlin / AWQ. *(Status: Mathematically Verified)* |
| **Inline SIMD FP16 SiLU Derivative** | Evaluates $\text{SiLU}'(x)$ inline via `__hfma2` intrinsics at output of backward GEMM. | **$2.0\times$ math throughput**; zero DRAM read/write roundtrip for $\nabla Y$. | **Standard Practice**: Epilogue fusion concept as implemented in CUTLASS / cuBLAS. *(Status: Analytically Modeled)* |
| **8B Model Fine-Tuning in 5.48 GB VRAM** | QLoRA (NF4 4-bit Base) + Full Activation Checkpointing. | Fits 8B training into **34% VRAM**, leaving **10.5 GB free** for $B=4$ batch scaling. | **Analytical Budget**: Based on QLoRA (Dettmers et al., 2023) & Checkpointing (Chen et al., 2016). *(Status: VRAM Math Verified)* |
| **Persistent Grid Streaming (40 Blocks)** | 40 wave-locked persistent blocks with L2 atomic counter tile fetching. | **Zero wave-tail waste**; flat 62W power profile. | **Established Practice**: Standard persistent grid pattern (Triton / CUTLASS) tuned for T4 40 SMs. *(Status: Analytically Modeled)* |

---

## Codebase Architecture & File Layout

```
research/
├── src/
│   ├── t4_cuda_kernels.cu             # Complete production CUDA C++ / PTX kernel suite
│   ├── t4_ptx_assembly_suite.cu       # Inline PTX header suite (LOP3, ldmatrix, mma.sync)
│   ├── t4_microbenchmarks.cu          # Standalone CUDA %clock64 timer micro-benchmark harness
│   └── t4_roofline_and_kernel_benchmarks.py # Roofline analyzer and empirical simulator
├── literature/
│   ├── ptx_sass_assembly_deep_dive.md # SASS disassembly (HMMA.884, UR0-UR63, control codes)
│   ├── t4_hardware_benchmarking_report.md # L1/L2/DRAM latencies & SMEM bank conflict profiles
│   ├── extreme_sass_and_ptx_microarchitecture.md # Sub-core dual-issue scheduling rules
│   ├── extreme_memory_cache_and_vram_architecture.md # GDDR6 BL=16 timing & SMEM swizzle math
│   ├── t4_training_gemm_research.md   # Forward, Backward dW, Backward dX 3-pass analysis
│   ├── t4_fused_optimizer_training_research.md # Fused AdamW & QLoRA VRAM scaling math
│   ├── audit_hardware_and_assembly_issues.md # Critical hardware audit & explicit remedials
│   └── audit_memory_and_precision_issues.md  # Critical precision/VRAM audit & remedials
├── to_human/
│   └── t4_cuda_research_presentation.html    # Interactive HTML presentation report
├── NEXT_STEPS.md                       # Roadmap for PyTorch extension & Colab integration
└── research-state.yaml                 # Central state manifest
```

---

## Quickstart & Compilation

To compile and execute the hardware micro-benchmarks on a Tesla T4 GPU:

```bash
# Compile micro-benchmarks with nvcc for sm_75
nvcc -O3 -arch=sm_75 research/src/t4_microbenchmarks.cu -o t4_microbenchmark

# Run empirical latency & clock throttling benchmarks
./t4_microbenchmark
```

To compile the CUDA C++ kernel suite:

```bash
nvcc -O3 -arch=sm_75 -c research/src/t4_cuda_kernels.cu -o t4_cuda_kernels.o
```

---

## Target Hardware Latencies (To Be Measured via `%clock64`)

- **L1 Data Cache Hit Latency**: ~28–32 cycles (Expected with `cudaFuncCachePreferL1`)
- **L2 Cache Hit Latency**: ~190–220 cycles
- **GDDR6 DRAM Read Latency**: ~400–450 cycles
- **Shared Memory Stride 32 (32-Way Bank Conflict Penalty)**: $32\times$ stall cycle serialization

---

## License

This project is released under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
