# H17 Mega-Kernel Benchmarking Plan

Source: analysis of `research-state.yaml` v9.1.0, `findings.md` §7, `src/kernels/h17_mega_kernel.cu`, `harness/empirical/run_empirical.py`, `tests/test_h17_fused_int3_gemv.py`.

---

## ⚠️ Pre-Benchmark Blockers (**FIXED** 2026-07-31 — see git)

1. **The "warp-specialized" kernel isn't.** `fused_h17_gemv_s3_warp_specialized_kernel` is a flat 256-thread uniform kernel — every thread does producer(dequant)+consumer(FMA) work. No `__syncthreads`-gated SMEM double-buffering, no producer/consumer split. H8 is analytically verified but **never hardware-measured**. Benchmarking this version shows kernel numbers, *not* the H8 contribution.
2. **Packing layout is disconnected from the test harness.** ~~`tests/test_h17_fused_int3_gemv.py` feeds zeros (`W_u32_np = np.zeros(...)`) to `fused_h17_gemv_s3`. The Python simulator packs 8×int3 per 3-byte triplet while the CUDA kernel unpacks 10×int3 per uint32 — the correctness gate on GPU is currently vacuous.~~ **FIXED 2026-08-01**: `test_h17_gpu_extension` now uses canonical 10-per-uint32 packing, per-group (group=100) scales/zp, deterministic non-zero data, a CPU reference GEMV (FP16 dequant + FP32 accumulate), and a `max_abs_diff <= 2.0` value gate + vacuity guard. The 8-per-3-byte `INT3Quantizer` class remains for the CPU-only simulation tests; the on-GPU gate uses the new canonical helpers.
3. **Per-column scales ≠ per-group scales.** Kernel signature takes `scale[N]` (one per column) but the quantizer uses block_size=128 groups per column. Confirmed mismatch.

---

## Tier 0 — Correctness gates (must PASS before perf numbers are publishable)

| Metric | Predicted | Method |
|---|---|---|
| KAT int3 pack→dequant→GEMV | max_abs_diff = 0.0 (bit-exact) | deterministic vector through `t4_kernels.fused_h17_gemv_s3` vs CPU reference |
| Random 4096×4096 GEMV vs PyTorch matmul | max_abs_diff ≤ 2.0 (FP16 accumulation envelope) | same gate as `fused_w4a16_gemm` in `expected.yaml` |
| Packing layout round-trip | 10×int3/uint32 layout == Python quantizer output | fix harness alignment first |
| Numerical residual annotation | ≤ 0.03 FP16 double-rounding (bounded) | document like H4's 0.0313721 |

---

## Tier 1 — Headline claim (research-state.yaml line 98)

> **2.5–4.5× decode speedup vs NF4 BitsAndBytes baseline on T4**

Required comparisons at **M=1, K=4096, N=4096** (decode regime, batch=1):

1. **Baseline A: BitsAndBytes NF4 dequant+matmul** (simulated NF4 path as in test file, plus actual `bitsandbytes` if available on Colab)
2. **Baseline B: llama.cpp CUDA INT3 path** — *mandated by novelty audit* (`findings.md` line 220: "New baseline requirement: llama.cpp CUDA INT3 path, not just BitsAndBytes NF4"). Reviewers will ask.
3. **Roofline reference: measured DRAM GB/s during kernel run** (target ≥303.4 GB/s = 94.8% of 320 GB/s — same standard H7 was held to)

Sweep the decode-relevant grid the test file already defines: **B ∈ {1,4,16} × M ∈ {1,128,2048}**.

---

## Tier 2 — Compile/static audit (gate via `expected.yaml`)

Add H17 entry with the same criteria H4 passed:

- `registers_per_thread ≤ 64` (H4 hit 30; H17 uses SMEM reduction so expect higher — flag if >64)
- `stack_spills_bytes = 0`
- presence of `lop3.b32` with `0x64046404` in PTX (confirms LUT 0x78/0x6A path fires, not a fallback)

---

## Tier 3 — H8 warp-specialization ablation (paper's core mechanical claim)

H17 folds in **two** independent contributions (H7's LOP3 dequant + H8's producer/consumer split-k). A controlled ablation attributes speedup correctly:

| Variant | What it isolates |
|---|---|
| **V0: naive** — dequant via `bfe.u32` + uniform GEMV | baseline instruction count |
| **V1: H7-only** — LOP3 dequant, no warp spec (≈ current `.cu`) | isolates warp-spec contribution |
| **V2: H17 full** — true producer/consumer double-buffered SMEM | isolates fusion |
| **V2b: H17 + H25 constant-bank scales** | the folded micro-technique ablation (audit explicitly requires it: "constant-cache broadcast vs SMEM scale-load ablation") |

Metrics per variant: wall-clock µs/token at M=1, DRAM bytes moved (~5.33× less than FP16), SM stall cycles (from `ncu`), L1/SMEM bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu.sum`, target 0).

---

## Tier 4 — Power/thermal (mandatory given findings §7.E)

Run under the same `nvidia-smi -lms 20` telemetry protocol as the 2026-07-30 run:

- Confirm stays under **70W TDP** at the kernel's natural occupancy (H17's 256-thread blocks on 40 SMs — check whether H5's 25% capping applies or if decode is naturally power-safe)
- Report mean SM clock + throttle_reason; the T4 throttling at 73.53W / 1222 MHz mean is the load-bearing motivation for H5, so H17 must not regress this

---

## Tier 5 — ncu Speed-of-Light (SKIP-if-absent, already supported by harness)

- `dram__throughput.of_peak_allocation_sol` ≥ 0.948 (the H7 standard)
- `l1tex__data_bank_conflicts_pipe_lsu.sum = 0`
- `smsp__pipe_tensor_op_hmma_cycles_active` — for the eventual WMMA consumer version (currently kernel has no WMMA; when added per findings §A, this becomes the SOL check)

---

## Recommended changes to harness

1. Add `H17_fused_int3_gemv` to `harness/empirical/expected.yaml` with `capability: kernel_ready` once blockers fixed. Metrics:
   - `decode_speedup_vs_nf4 (gte 2.5)`
   - `decode_speedup_vs_llamacpp_int3 (informational but report)`
   - `dram_sol_pct (gte 0.948)`
   - `max_abs_diff (lte 2.0)`
   - `registers_per_thread (lte 64)`
   - `stack_spills_bytes (lte 0)`
2. Add `stage_h17_ablation` to `run_empirical.py` that runs V0/V1/V2/V2b and emits a per-variant table — this is what goes in the ASPLOS paper's evaluation section.
3. Fix the packing-layout contract: one canonical int3 layout (recommend the kernel's existing 10-per-uint32 since it's already SASS-verified in `lop3_dequant.h`), update the Python quantizer to match, and use non-zero deterministic data in `test_h17_gpu_extension`.

---

**Bottom line:** benchmark what proves the 2.5–4.5× claim against NF4 *and* llama.cpp INT3, at M=1/K=4096/N=4096, with the V0–V2b ablation ladder so reviewers can see exactly how much comes from LOP3 dequant vs warp specialization vs constant-bank scale streaming. Right now the kernel has only the LOP3 part — first benchmark run will likely land at the *bottom* of the 2.5–4.5× range; the H8 producer/consumer rewrite is what closes the gap.
