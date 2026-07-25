# Strategic Impact & Technical Analysis: Tesla T4 (Turing CC 7.5) Inference & Training

This document provides a grounded technical analysis of microarchitectural optimization strategies for both **Inference** and **Training/Fine-Tuning** on **NVIDIA Tesla T4 GPUs (Turing CC 7.5, 40 SMs, 70W TDP, 16 GB GDDR6)**, providing prior art attributions, analytical models, and explicit empirical test plans.

---

## Executive Summary & Contribution Framing

To maintain academic and technical clarity, techniques analyzed in this document are framed into two distinct categories:

1. **Original Research Contributions**:
   - **Power-Aware Occupancy Cap & Regime Split**: Capping occupancy at 25% for compute-bound prefill ($M\ge 2048$) to lock 1590 MHz boost clock under T4's 70W TDP ceiling, while relaxing occupancy to 50%–75% for memory-bound decode ($M=1$) to maximize Memory Level Parallelism (MLP).
   - **Fused Backward GEMM + AdamW Register Accumulation**: Fusing weight gradient accumulation ($\nabla W$) across K-loops and executing AdamW parameter updates directly inside register fragments before write-back to DRAM.

2. **Prior Art Integrations & Standard Practice**:
   - **LOP3 FP16 Magic Exponent Dequantization**: Integration of established community techniques (**ExLlama / ExLlamaV2**, **Marlin**, **AWQ**) to Turing CC 7.5.
   - **Persistent Grid Streaming & Epilogue Fusion**: Application of standard CUDA patterns (**cuBLAS**, **CUTLASS**, **Triton**) tuned for T4's 40 SMs.
   - **QLoRA + Checkpointing VRAM Modeling**: Memory budget modeling derived from published literature (**Dettmers et al., 2023**; **Chen et al., 2016**).

---

## Part A: INFERENCE Optimization Analysis

### 1. Sub-Byte Unpacking & Dequantization Integration
- **INT4 Dequantization via Magic Exponent Insertion**:
  - *Context*: Unpacks 4-bit weights into FP16 registers using `lop3.b32` with magic exponent `0x6400` (prior art: ExLlamaV2/Marlin/AWQ).
  - *Target*: Bypasses integer-to-float conversion pipelines. Signed INT4 variant (`lop3.b32` LUT `0x6A`) inverts sign bit 3 in a single cycle.

### 2. Autoregressive Decode Occupancy Management ($M=1$)
- **Regime-Split Power & Occupancy Management (Original Contribution)**:
  - *Mechanism*: Automatically scales occupancy to **50%–75% during decode ($M=1$)**. Because Tensor Cores spend $>90\%$ of cycles waiting for KV cache vectors ($\alpha < 0.10$), active warp compute power drops to $<0.02\text{W}$, allowing higher occupancy without exceeding 70W board TDP.
  - *Target*: Maximizes Memory Level Parallelism (MLP) to saturate GDDR6 bandwidth during single-token decoding.

---

## Part B: TRAINING & FINE-TUNING Optimization Analysis

### 1. Power-Aware Occupancy Cap for Compute-Bound Backward Passes ($M \ge 2048$)
- **Mechanism**: During heavy matrix multiplications ($dW$ and $dX$ backward GEMMs), capping thread occupancy at **25.0% (8 warps / 256 threads per SM)** maintains active board power under 70W.
- **Predicted Benefit**: Prevents NVPM thermal clock downclocking, sustaining 1590 MHz boost clock throughout multi-hour training runs.

### 2. Fused Backward GEMM + AdamW (Original Contribution)
- **Optimizer Memory Traffic Reduction**:
  - *Mechanism*: Standard training loads weights $W$, reads gradients $\nabla W$, reads first momentum $m$, reads second momentum $v$, updates states, and writes back to DRAM (28 Bytes/param).
  - *Analytical Model*: Accumulates $\nabla W$ in register fragments across the $K$-loop, then fuses the AdamW weight update directly in registers before writing back to global memory (22 Bytes/param, a predicted 21.4% memory traffic reduction).

### 3. VRAM Budget Modeling: 8B Model Fine-Tuning
- **Analytical Memory Profile (QLoRA + Full Checkpointing)**:
  - *Base Model (NF4 4-bit)*: ~3.50 GB VRAM.
  - *LoRA Adapters ($r=16$)*: ~0.12 GB VRAM.
  - *Optimizer States (FP32 AdamW for Adapters)*: ~0.48 GB VRAM.
  - *Activation Memory (Full Checkpointing)*: ~1.38 GB VRAM.
  - *Total Predicted Peak*: **~5.48 GB VRAM** (out of 16 GB on T4).

---

## Technical Status & Empirical Verification Plan

All performance numbers in this document are **analytically derived targets**. Empirical benchmarking against established baselines (**TensorRT**, **ExLlamaV2**, **llama.cpp**, **vLLM**) on physical T4 hardware is required to validate achievable throughput gains.
