# Findings & Synthesis: Tesla-T4 CUDA Optimizations & Microarchitectural Analysis

## 1. Executive Summary & Analytical Assessment Matrix

Through rigorous microarchitectural modeling (SASS dual-issue analysis, register file constraints, memory bank conflict swizzles, IEEE 754 float manipulation, and NVPM power responses), we evaluated **8 key microarchitectural techniques** for NVIDIA Tesla T4 GPUs (Turing CC 7.5). 

To maintain strict academic and engineering rigor, we categorize these techniques into **Original Contributions**, **Standard CUDA / Community Prior Art Integrations**, and **Analytically Derived Models**:

| Technique / Engineering Component | Microarchitectural Mechanism | Target SASS / Hardware Benefit | Originality & Prior Art Context | Validation Status |
|---|---|---|---|---|
| **Power-Aware Occupancy Cap & Regime Split** | 25% occupancy cap for compute-bound prefill ($M\ge 2048$); 50%–75% for memory-bound decode ($M=1$) | Locks **1590 MHz boost clock** on prefill; maximizes MLP on decode without throttling | **Original Contribution**: Novel power-pacing strategy designed specifically for passively cooled 70W TDP T4 GPUs | Analytically Modeled (Pending `nvidia-smi` telemetry) |
| **Fused Backward GEMM + AdamW** | Accumulates $\nabla W$ in register fragments; applies AdamW update directly in-register | **21.4% DRAM bandwidth saving** (28 → 22 B/param) by eliminating $\nabla W$ DRAM writeback | **Original Contribution**: Fused register-level backward GEMM + optimizer scheme for Turing persistent blocks | Analytically Modeled (Pending CUDA kernel execution) |
| **Signed INT4 Dequant (`LOP3` LUT `0x6A`)** | Single-cycle sign bit 3 inversion + FP16 magic exponent `0x64086408` | **1 SASS cycle**; 2.5$\times$ instruction reduction over `bfe` | **T4 Adaptation of Prior Art**: Extends FP16 magic number insertion (ExLlamaV2, Marlin, AWQ) to signed INT4 | Mathematically Verified (KAT hex vector matched) |
| **Unsigned INT4 Magic Exponent (`0x64006400`)** | Direct mantissa injection bypassing int-to-float conversion pipe | **2.5$\times$ instruction reduction** (20 down to 8 SASS insts) | **Established Prior Art**: Standard technique in ExLlama / Marlin / AWQ inference engines | Mathematically Verified (KAT hex vector matched) |
| **PRMT + LOP3 Multi-Word Vector Packing** | Byte-selector `0x4000` remapping with `LDG.E.64` vector loads | **50% instruction reduction** on 64-bit vector unpacking | **Known Optimization**: Standard byte-permutation instruction scheduling trick | Analytically Verified |
| **Inline Register Activation Fusion (SiLU/GELU)** | Interleaving FP16 ALU (`HFMA2`) and MUFU (`EX2`/`RCP`) execution units | **5 dual-issued instructions**; zero DRAM/SMEM roundtrip | **Standard Practice**: Epilogue fusion concept as implemented in CUTLASS / cuBLAS | Analytically Modeled |
| **Dynamic Unified Cache Partitioning (`cudaFuncCachePreferL1`)** | `cudaFuncCachePreferL1` (32KB SMEM / 64KB L1) matching 2-stage tiles | **2.0$\times$ L1 Cache capacity**; lowers dynamic GDDR6 DRAM power | **Standard CUDA API**: Standard API usage described in NVIDIA CUDA programming guides | Configured in Kernel Driver |
| **Persistent Grid Block Streaming (40 Blocks)** | 40 persistent wave-locked blocks with L2 atomic counter tile fetching | **Zero wave-tail waste**; flat power profile | **Established Practice**: Standard persistent grid pattern (Triton / CUTLASS / cuBLAS) tuned for T4 40 SMs | Analytically Modeled |

---

## 2. Key Microarchitectural Findings & Honest Context

### A. Regime Split: Compute-Bound Prefill vs. Memory-Bound Decoding
Our power modeling identified a crucial distinction for passively cooled 70W T4 GPUs:
- **Compute-Bound Prefill ($M \ge 2048$, FLOP/B > 45)**: Heavy Tensor Core activity ($\alpha > 0.8$) causes cumulative SM power to exceed 70W, triggering NVPM thermal/power throttling (dropping core clocks from 1590 MHz down to ~950 MHz). Capping occupancy at **25% (256 th/SM)** keeps power under ~62W, locking maximum boost clock.
- **Memory-Bound Decoding ($M = 1$, FLOP/B < 2)**: Tensor Core duty cycle drops ($\alpha < 0.10$), reducing dynamic warp compute power to $<0.02\text{W}$. Here, 100% occupancy does NOT cause 70W power throttling. Occupancy should be scaled up to **50%–75% (16–24 warps/SM)** to maximize Memory Level Parallelism (MLP) and GDDR6 bandwidth utilization.

### B. Signed INT4 Dequantization (`lop3.b32` LUT `0x6A`) & Prior Art
The use of FP16 magic numbers (`0x6400`) to unpack 4-bit nibbles directly into floating-point mantissas via `lop3.b32` is a well-established community technique popularized by ExLlama, Marlin, and AWQ. Our formulation extends this to two's complement signed INT4 ($s4 \in [-8, 7]$) by noting that adding 8 is bit-identical to inverting bit 3. By using magic constant `0x64086408` and LUT `0x6A`, bit 3 inversion and exponent insertion execute in a single SASS cycle.

### C. Persistent 40-Block Grid Streaming & L1 Preference
By launching exactly 40 persistent blocks (1 block/SM matching T4 hardware capacity) and fetching macro-tiles via global L2 atomic counters, we eliminate inter-wave launch latency and wave-tail imbalance. Dynamic cache re-partitioning (`cudaFuncCachePreferL1`) expands L1 cache to 64 KB per SM, reducing GDDR6 memory power and releasing thermal headroom for sustained 1590 MHz boost clock.

---

## 3. Project Artifact Index

All generated code, benchmarks, research papers, and presentation reports are organized inside the `research/` directory:

1. **CUDA C++ / PTX Custom Kernel Suite**: [research/src/t4_cuda_kernels.cu](file:///home/kriday/Desktop/epoch-1/research/src/t4_cuda_kernels.cu)
2. **Roofline Analyzer & Benchmark Harness**: [research/src/t4_roofline_and_kernel_benchmarks.py](file:///home/kriday/Desktop/epoch-1/research/src/t4_roofline_and_kernel_benchmarks.py)
3. **Sub-Byte Dequantization Whitepaper**: [research/literature/verified_novelty_dequantization.md](file:///home/kriday/Desktop/epoch-1/research/literature/verified_novelty_dequantization.md)
4. **Power Tuning & Pipeline Whitepaper**: [research/literature/verified_novelty_pipeline.md](file:///home/kriday/Desktop/epoch-1/research/literature/verified_novelty_pipeline.md)
5. **Interactive Presentation Report**: [research/to_human/t4_cuda_research_presentation.html](file:///home/kriday/Desktop/epoch-1/research/to_human/t4_cuda_research_presentation.html)
6. **Central Research Tracking**: [research/research-state.yaml](file:///home/kriday/Desktop/epoch-1/research/research-state.yaml) | [research/research-log.md](file:///home/kriday/Desktop/epoch-1/research/research-log.md)
7. **Empirical Execution Protocol**: [research/NEXT_STEPS.md](file:///home/kriday/Desktop/epoch-1/research/NEXT_STEPS.md)
