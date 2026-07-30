# Empirical Validation Harness — T4 CUDA Research (H4 / H7 / H9 family)

This package converts the repo's **simulation-verified** hypotheses into
**hardware-measured evidence** on a physical Tesla T4 GPU, producing a
persistent, committable paper trail under `research/results/<timestamp>/`.

---

## TL;DR

1. Open the notebook on a Colab T4: [`colab_run_empirical.ipynb`](../../colab_run_empirical.ipynb)
2. Runtime → Change runtime type → **T4 GPU** → **Run all**
3. It auto-downloads `results/<ts>.tar.gz`
4. Extract that into your local `research/results/`, `git add research/results/<ts>`, commit
5. Paste the resulting `VERDICT.md` + `summary.json` back to the agent to update statuses

---

## What it actually measures (and what it honestly cannot)

The key design principle: **distinguish measurable claims from sim-only claims.**

| Hypothesis | Real CUDA kernel? | What the harness does |
|---|---|---|
| **H4 signed INT4 LOP3 (0x6A)** | ✅ `dequantize_s4` | KAT bit-exactness on GPU, 4096-word random allclose, ptxas regs/spills |
| **H4 unsigned INT4 LOP3 (0xEA)** | ✅ `dequantize_u4` | KAT bit-exactness on GPU, 4096-word random allclose, ptxas regs/spills |
| **Fused W4A16 GEMM** | ✅ `fused_w4a16_gemm_u4/s4` | Max-abs-diff vs PyTorch `matmul` reference |
| **H7 INT3 LOP3 (0xCA)** | ❌ **no kernel yet** | Runs the 8-state math-identity check, then reports **NOT_MEASURABLE** |
| **H9 FP8 E4M3→FP16 (0xEA re-bias)** | ❌ **no kernel yet** | Runs the 254-state E4M3 sweep, then reports **NOT_MEASURABLE** |

The last two rows are deliberate: the repo currently has **no INT3 or FP8 CUDA
kernel** (only INT4). H7/H9 headline numbers (13-instr dequant, 60.1 TFLOPS) are
simulation claims; the harness says so rather than pretending to measure them.
Claims about INT3/FP8 *throughput* stay `SIMULATED` until those kernels are written.

## Stages

| Stage | Needs | Output |
|---|---|---|
| S0 env stamp | — | GPU/driver/CUDA/torch/git SHA recorded |
| S1 compile audit | nvcc | `ptxas -v` register counts + spill bytes per kernel |
| S2 build ext | nvcc + torch | `t4_kernels*.so`, build log |
| S3 GPU correctness | CUDA GPU | KAT + random allclose + fused GEMM diff |
| S4 telemetry | `nvidia-smi` + GPU | `telemetry.csv` (power/clock/throttle @ 20 ms) |
| S5 ncu SOL | `ncu` + GPU | `ncu_summary.csv`; **SKIP on Colab free** |
| S6 verdict | — | `VERDICT.md` + `summary.json` + tarball |

Each metric from [`expected.yaml`](expected.yaml) is evaluated with a criterion
(`gte` / `lte` / `eq` / `bool`) → **CONFIRM / MARGIN / FAIL / NOT_MEASURABLE**.

## Result bundle layout

```
research/results/2026-08-01_153012/
├── run.log                # full stdout/stderr, teed
├── summary.json           # machine-readable: env, metrics, verdicts
├── VERDICT.md             # human-readable per-hypothesis table  ← paste to agent
├── compile_audit.json     # parsed ptxas info
├── build_log.txt          # extension build output
├── telemetry.csv          # nvidia-smi samples
└── ncu_summary.csv        # only if ncu was available
```

## Exit code & global verdict

- `0` → every **kernel_ready** hypothesis CONFIRMs (or is only MARGIN)
- `1` → at least one kernel_ready hypothesis FAILs
- `sim_only` hypotheses (H7, H9) never block the run — they are recorded as
  `NOT_MEASURABLE`, which on paper becomes the known-limitations note.
- Global verdict is one of `CONFIRM` / `MARGIN` / `FAIL` / `ERROR`.

## Running without Colab (rented T4: RunPod / Lambda)

```bash
git clone <your-fork> /workspace/repo && cd /workspace/repo/research
pip install pyyaml torch
python3 harness/empirical/run_empirical.py            # full run incl. ncu if present
# or targeted:
python3 harness/empirical/run_empirical.py --skip-ncu --skip-telemetry
```

## Extending expected.yaml

Add a metric under the right hypothesis with `predicted`, `unit`, `criterion`,
`tolerance`. Statuses flow automatically into `VERDICT.md`. Mark a hypothesis
`kernel: sim_only` until its CUDA kernel exists — that's the audit trail.

## Design choices worth knowing

- **No PyYAML-on-bare-box fallback.** The script fails loudly with an install hint
  rather than silently mis-parsing. (Colab ships PyYAML by default.)
- **Warp-specialization (H8 / H5) not in this scope.** They need kernel-side
  instrumentation + thermal loop, not just dequant. This package covers the
  LOP3 dequant family only, as agreed.
