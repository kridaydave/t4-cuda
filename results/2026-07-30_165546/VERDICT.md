# Empirical Validation Verdict — 2026-07-30_165546

- Timestamp (UTC): 2026-07-30T16:57:21.374694+00:00
- Git SHA: unknown  (dirty: True)
- Device: Tesla T4
- CUDA: 12.8  | PyTorch: 2.11.0+cu128

## GLOBAL VERDICT: **FAIL**

## Per-hypothesis results

| Hypothesis | Capability | Verdict |
|---|---|---|
| H4_signed_int4_lop3 | kernel_ready | FAIL |
| H4_companion_unsigned_int4_lop3 | kernel_ready | FAIL |
| fused_w4a16_gemm | kernel_ready | CONFIRM |
| H7_signed_int3_lop3 | sim_only | NOT_MEASURABLE (sim_only) |
| H9_fp8_lop3_rescale | sim_only | NOT_MEASURABLE (sim_only) |

## Stage status

| Stage | Status | Duration (s) | Notes |
|---|---|---|---|
| S0_env | PASS | 1.7 |  |
| S1_compile_audit | PASS | 3.9 |  |
| S2_build_ext | PASS | 44.3 | built .so files: ['t4_kernels.cpython-312-x86_64-linux-gnu.so'] |
| S3_gpu_correctness | FAIL | 16.7 | H7 INT3: validated math identity ONLY — kernel is sim_only (not measurable).; H9 FP8: validated reference sweep ONLY — k |
| S4_telemetry | PASS | 8.4 |  |
| S5_ncu | PASS | 19.9 |  |
| S6_verdict | FAIL | 0.0 |  |

## Metric detail

| Metric | Predicted | Measured | Status |
|---|---|---|---|
| H4_signed_int4_lop3.registers_per_thread | 64 | 30 | CONFIRM |
| H4_signed_int4_lop3.stack_spills_bytes | 0 | 0 | CONFIRM |
| H4_companion_unsigned_int4_lop3.registers_per_thread | 64 | 30 | CONFIRM |
| H4_companion_unsigned_int4_lop3.stack_spills_bytes | 0 | 0 | CONFIRM |
| H4_companion_unsigned_int4_lop3.kat_bit_exact | 0.0 | 2 | FAIL |
| H4_signed_int4_lop3.kat_bit_exact | 0.0 | 4 | FAIL |
| H4_signed_int4_lop3.random_allclose | 0.0 | 0.03125 | FAIL |
| H4_companion_unsigned_int4_lop3.random_allclose | 0.0 | 0.031372 | FAIL |
| fused_w4a16_gemm.max_abs_diff | 2.0 | 1.5879 | CONFIRM |
| fused_w4a16_gemm.correctness_u4 | True | 1 | CONFIRM |
| fused_w4a16_gemm.correctness_s4 | True | 1 | CONFIRM |
| H7_signed_int3_lop3.kat_bit_exact_int3 | 0.0 | 0 | CONFIRM |
| H9_fp8_lop3_rescale.fp8_sweep_valid_states | 254 | 254 | CONFIRM |
| telemetry.power_within_tdp | 70.0 | 73.53 | FAIL |
| telemetry.sm_clock_mean | 1590.0 | 1222.2 | INFO |

---

Artifacts in this directory: `run.log`, `summary.json`, `telemetry.csv`, `compile_audit.json`, `build_log.txt`, `ncu_summary.csv` (if ncu ran).
