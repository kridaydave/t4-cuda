# Literature: 2026 Mid-Year Research Sweep — Reading Between The Lines
## Autoresearch Outer Loop — 2026-07-29 (supersedes/extends survey_2026_07_26)

Three parallel scouts swept arXiv/GitHub 2025-2026 across three axes: (A) kernels & quantization,
(B) speculative decode & KV residency, (C) persona steering & eval. This file records what they
found, what it does to our novelty claims, and the gaps that become the H26+ generation.

---

## PART 1 — NOVELTY RE-AUDIT OF EXISTING HYPOTHESES

### Kernel / Quantization axis (H4/H7/H8/H9/H17/H23/H25/H5)

| Hypothesis | Old Status | 2026-07-29 Verdict | Evidence Summary |
|---|---|---|---|
| H7 (INT3 LOP3 LUT 0xCA) | Claimed "first single-cycle INT3 LOP3" | **PARTIALLY-SCOOPED (folklore) / STILL PUBLISHABLE** | LOP3 + magic-exponent dequant is entrenched in production code (FasterTransformer, AWQ/TRT-LLM, Marlin) but is **folklore — nobody has ever characterized it in a paper**. INT3-specific LOP3 LUTs (no nibble alignment) with measured SASS instruction counts remain unclaimed. Publishable as a SASS-level catalog + INT3 extension, NOT as first-invention. |
| H8 (software warp specialization, pre-Ampere) | Claimed unique | **NOVEL — and value went UP** | All 2025-26 warp-spec work (Tawa CGO'26, Cypress, Sim-FA, SM90 SwiGLU pingpong) presumes Hopper TMA/wgmma async hardware. Software-emulated producer/consumer on sm_75 is unclaimed. Frame as "warp specialization without async hardware." |
| H9 (FP8 E4M3 → FP16 TC via LOP3 re-bias, LUT 0xEA) | Claimed original | **NOVEL** | Zero hits for FP8 emulation on legacy tensor cores; FP8 literature assumes Ada/Hopper native support. Defense burden: document where the +8 exponent re-bias is exact vs E4M3 denormals/NaN — reviewers will attack numerics. |
| H23 (1.58-bit ternary bit-serial LOP3) | Claimed novel | **NOVEL (GPU) / concept partially scooped (CPU)** | bitnet.cpp (2410.16144) and T-MAC (EuroSys'25, 2407.00088) do ternary via **LUT on CPU**; Intel (2508.06753) does 2-bit GEMM on **Xe** GPUs. **No CUDA ternary bit-serial popcount kernel exists.** Differentiator vs T-MAC: compute-by-logic-ops instead of compute-by-lookup (SMEM LUT bandwidth is scarce on GPU). |
| H25 (constant-bank scale streaming) | Claimed novel | **NOVEL but thin** | A code-level trick, not a paper-level claim. **RECOMMEND: fold into H17 as a measured micro-technique** (constant-cache broadcast vs SMEM scale loads) rather than a standalone hypothesis. |
| H5 (70W occupancy-as-clock-controller) | Claimed original | **NOVEL (specific mechanism)** | Power capping generally is well-trodden; **occupancy-as-DVFS-controller for LLM decode on 70W passively-cooled parts** is unclaimed. Must verify against vGPU/time-slicing confounds and document the DVFS hysteresis band. Empirical-systems claim; pair with perf/W curves. |
| H17 (fused INT3 dequant + warp-spec GEMV mega-kernel) | Proposed | **NOVEL — strongest flagship claim** | The fused-dequant-GEMV-decode search space is empty pre/post Ampere for this quantization regime. **Baseline warning from scout: also compare llama.cpp CUDA INT3 path, not just BitsAndBytes NF4** — reviewers will ask. |

**Overall: nothing in 2025-26 literature supersedes any claim.** The sm_75 niche is genuinely
uncrowded — the field's attention has moved wholesale to Hopper/Blackwell async hardware.

### Serving / Speculative-Decode axis (H20/H22/H24)

| Hypothesis | Old Status | 2026-07-29 Verdict | Evidence Summary |
|---|---|---|---|
| H20 (INT3 0.5B draft → 32B target, batch=1) | Proposed | **PARTIALLY-SCOOPED — needs reframing** | Quantized-draft SD is now published: ML-SpecQD (MXFP4 drafts, 2503.13565), QuantSpec (4-bit self-draft, >90% acceptance, 2502.10424), SPEQ (draft from target's own weight bits, 2510.18525), Quasar (quantized verification, 2603.01399). **The open gap:** extreme (INT3) quantization of a *mismatched-size* draft, measured as acceptance-rate cost vs bandwidth savings, batch-1, on legacy bandwidth-bound GPU. Reframe as that study + system, not "quantized draft SD" generically. |
| H24 (KV cache in thread-block registers, S≤128) | Simulated/proved | **NOVEL (per sweep)** | No published register-resident KV found anywhere. Closest neighbors are SMEM-level fused decode and LLC/MSHR work (LLaMCAT). Caveat: 2-query negative result; verify against FlashInfer/TRT-LLM issues and NanoFlow/Teola persistent-kernel serving papers before public novelty claim. |
| H22 (INT3/INT4 QAT for Turing) | Proposed | **PARTIALLY-SCOOPED concept, open execution** | SpinQuant/BitsMoE/MR-GPTQ (2509.23202: Hadamard-rotation-fused MXFP4) own the QAT-for-low-bit concept. Open: QAT *targeting Turing's LUT-dequant datapath* specifically, and QAT loss against **acceptance rate** (not perplexity) for drafts. |

### Persona / Eval axis (H19/H21)

| Hypothesis | Old Status | 2026-07-29 Verdict | Evidence Summary |
|---|---|---|---|
| H19 (activation-steering persona vectors) | Proposed | **PARTIALLY-SCOOPED — white space identified** | GCAD (2605.10664) now has hard multi-turn numbers (turn-10 trait expression 78.0→93.1; coherence drift −18.6→−1.9) — the generic claim is done. PSR/Steer-Like-the-LLM (ICML'26, 2605.03907) shows token-uniform steering is mechanistically unfaithful vs prompting. **Still-open white space:** (a) principled multi-persona coexistence (Creative Collision 2606.16240 documents dominance/collapse, no fix exists); (b) steering on INT3/INT4-quantized persona models — **literally zero papers** (see Part 2, item 1). |
| H21 (format-bleed + persistence eval) | Proposed | **PARTIALLY-SCOOPED (persistence half) / NOVEL (format-bleed half)** | ContextEcho (2605.24279) is a ready-made deployment-scale drift harness (25-probe identity suite, 23 models, 3.7k-9.7k turn sessions); it even *observes* drift breaking formatting contracts but never operationalizes a format-bleed rate. **"Format bleed / format disambiguation" as named constructs return zero arXiv hits — our coinage.** Adopt standard metrics (below) for persistence; claim format-bleed as first-class metric. |

**Standard persona-persistence metrics to ADOPT for H21 (instead of inventing):**
- Turn-N trait expression (% responses expressing trait at turn N) — GCAD: turn-10, 78.0→93.1
- Coherence drift (Δ judged coherence turn 1→N) — GCAD: −18.6→−1.9
- Assistant-Axis deviation / persona-vector projection distance (Lu et al. 2601.10387)
- Prompt-to-line / line-to-line / Q&A consistency (Abdulhai et al. 2511.00222, human-validated)
- PVNI neutral-interpolation scores for prompt-insensitive eval (2601.09833)
- Embedding-anchor cosine drift, ROC AUC for drift detection (Nautilus Compass, 0.83 AUC, black-box)
- OOC atomic-level fidelity (Shin et al. 2506.19352)
- Echo/role-confusion rate + persona separability (SPASM, 2604.09212)

---

## PART 2 — READING BETWEEN THE LINES: THE NON-OBVIOUS FINDINGS

These are the discoveries that emerge from *juxtaposing* the three sweeps — things no single
paper says outright.

### 1. THE BIG ONE: Quantization × Activation-Steering is unstudied — and it is load-bearing for Epoch
ArXiv searches for (steering × INT3/INT4), (steering × weight quantization), (persona-vector ×
quantization) return **ZERO results**. Only two adjacent data points exist: HELIX (2602.17691,
manifold steering still works on 4-bit quantized models, sparse-layer steering suffices) and
GUARD-IT (2605.12765, activation steering "remains effective under quantization" while
weight-editing degrades). **Nobody has measured whether persona vectors extracted at FP16 survive
INT3 weight quantization, whether extraction must be redone on the quantized model, or how
quantization error perturbs vector direction (cosine drift).** This is not an academic nicety:
Epoch serves Eli/Theo/Albert INT3-quantized on T4s. If steering vectors silently break under INT3,
H19 fails for an invisible reason. If they survive with quantifiable margin, that measurement is
itself a contribution. **→ Becomes H26.**

### 2. The 2026 warp-specialization hype cycle *raises* the value of H8
Tawa (CGO'26), Cypress, Sim-FA, SM90 pingpong fusion — warp specialization is THE hot kernel
topic of 2026, and every paper assumes Hopper async hardware. The implicit consensus is "you need
TMA/wgmma to do this." A software-emulated producer/consumer path on sm_75 with 94.2% stall
reduction is now a *contrarian* result: "you don't need Hopper for this." Frame H8 accordingly.

### 3. The MXFP4 wave makes T4 LOP3 machinery suddenly load-bearing
gpt-oss ships MXFP4 weights; MR-GPTQ (Hadamard rotations fused into weights, E8M0 scales) is the
leading 2025-26 quant recipe; MicroMix (ICLR'26) does per-channel MXFP4/6/8 mixtures. E8M0 group
scales are power-of-two → **multiply = exponent add**, which is exactly our LOP3/IADD machinery.
The legacy-GPU answer to the MXFP4 wave (rotate weights offline, 4-bit LUT → FP16, exponent-add
scaling on Turing FP16 tensor cores) is unwritten. **→ H27.**

### 4. Ternary inference is CPU-centric — Microsoft left the CUDA gap open
bitnet.cpp (TL1/TL2 LUTs, CPU-only), T-MAC (CPU), Intel Xe2 2-bit GEMM (Intel silicon),
VibeVoice-ASR-BitNet (ggml SIMD, CPU, Jul 2026). The W1.58A4 serving regime (BitNet a4.8) has
**no CUDA kernel**. H23's sign/mask bitplane + popcount path is the natural GPU-native answer
(compute-by-logic-ops beats compute-by-lookup where SMEM LUT bandwidth is scarce). Completing
W1.58A4 CUDA (INT4 activations via H4's LUT × H23's ternary weights, popcount-with-nibble-operand
accumulation) closes the gap Microsoft/Intel left. **→ H29.**

### 5. Acceptance rate, not perplexity, is the right quantization loss for drafts
QuantSpec shows 4-bit self-drafts keep >90% acceptance; Quasar shows quantized *verification*
preserves accepted length. The implied-but-unstated corollary: **quantize the draft against
draft–target token agreement (top-k KL, EAGLE-distillation-style), not against draft perplexity.**
An INT3 draft trained against acceptance plausibly recovers the naive-W4A16 acceptance drop —
and the 8:1 draft/target weight-streaming gap is exactly where the 320 GB/s T4 wins. **→ folds
into reframed H20.**

### 6. Persona superposition fails *in documented ways*, and nobody has the fix
Creative Collision (2606.16240): two superimposed persona vectors — one can dominate/suppress the
other across the whole interpolation range. 53-trait audit (2607.13162): two steerable traits
composed can collapse; pairs involving a default never do. SPASM: LLM–LLM echoing is a failure
mode. **Serving Eli/Theo/Albert from shared vectors will hit all three.** Scheduled/orthogonalized
(SAS-style residualization) or per-turn routed multi-persona steering is unclaimed. **→ H31.**

### 7. Steering rots: fixed layer/coefficient configs decay under perturbation
Adversarial Robustness of Activation Steering (2606.07696): under input perturbation, directional
robustness drops ≤64% and **optimal layer shifts by up to 17 positions**. ASTEER (2606.11599):
steerability is predictable from early hidden states (0.7 macro-F1, 150 concepts). Together they
imply a deployment practice nobody has built: **re-scan layer choice per persona after any
model/template/quantization change**, using cheap early-hidden-state probes. Engineering-shaped,
directly serves the T4 pipeline. **→ H33.**

### 8. vLLM's own production study hands us the T4 roofline lever for speculative decode
The vLLM production SD study (2601.11580): verification dominates; acceptance length varies
strongly by position. On a bandwidth-bound batch-1 T4, a 32B forward is a **fixed-cost weight
stream** — the only lever is tokens-accepted-per-verification. A per-position online acceptance
model that dynamically reshapes the draft tree (depth/width per node) to maximize E[accepted
tokens] per verify pass — solving explicitly for the roofline crossover — is unclaimed
(EntMTP/DEL are adaptive but don't solve the legacy-bandwidth crossover). **→ H28.**

### 9. The taste axis is still vacant — but H18's death was correct
Beyond Resolved Rate (2607.18462): on SWE-bench Lite, newer models resolve more but show **no
significant improvement in non-functional patch quality**. Senior SWE-Bench's niche remains
unfilled; nobody productized taste. Consistent with H18's kill decision: taste stays human-curated.
No action — recorded for the record.

### 10. Compile-time warp-spec for pre-async GPUs is an open framework play
Inverted from Tawa (auto producer/consumer partitioning, TMA-only): an aref-lite lowering that
targets double-buffered SMEM + volatile flags + `__threadfence_block` for sm_75/sm_80 would turn
H8 from one hand-crafted kernel into an automatable schedule, with the 94.2% hand-tuned point as
motivation. High effort; flags a paper's systems-contribution path. **→ H30.**

---

## PART 3 — NEW HYPOTHESES REGISTERED (H26–H33)

(Full formulations in research-state.yaml. Summary:)

- **H26** Quantization-aware persona vector extraction/verification (FP16→INT4→INT3 stability;
  extract on deployment model if degraded). *De-risks H19. Zero prior work.*
- **H27** Rotation-fused MXFP4-W/FP16-A GEMV for T4 (offline Hadamard, 4-bit LUT dequant, E8M0
  exponent-add scaling). *Rides the gpt-oss MXFP4 wave onto legacy hardware.*
- **H28** Position-aware speculative-tree shaping for bandwidth-bound decode (online per-position
  acceptance model, dynamic tree depth/width vs fixed verify cost). *Reframes/complements H20.*
- **H29** W1.58A4 CUDA serving kernels (H4 INT4-activation LUT × H23 ternary bitplanes,
  popcount-with-nibble accumulation). *Completes the gap bitnet.cpp/T-MAC left on the table.*
- **H30** aref-lite: compiler-generated software warp specialization for pre-async GPUs
  (H8 as an automatable schedule). *Framework-scale, high effort.*
- **H31** Multi-persona coexistence without directional dominance (orthogonalized/routed steering
  for Eli/Theo/Albert; defeats Creative-Collision suppression + SPASM echoing).
- **H32** Format-bleed rate as a first-class persistence metric and *leading indicator* of drift
  (operationalizes what ContextEcho only observes; correlate against Assistant-Axis deviation).
  *Folds into H21 as the differentiator. We own the coinage.*
- **H33** Steerability regression probes for deployment (ASTEER-style early-hidden-state
  prediction + post-change layer re-scan, after model/template/quantization changes).

## PART 4 — CONSOLIDATION DECISIONS (executed 2026-07-29)

1. **H25 folded into H17** — demoted from standalone hypothesis to a measured micro-technique
   inside the mega-kernel (constant-bank vs SMEM scale-load ablation).
2. **H20 reframed** — from "INT3 draft speculative decoding" to "acceptance-first quantized
   drafting: INT3 acceptance-sweep + QAT-against-agreement on bandwidth-bound batch-1 decode."
3. **H19 de-risked by sequencing** — H26 (does steering survive INT3?) gates H19 engineering.
4. **H21 adopts standard metrics** (GCAD turn-N expression, Assistant-Axis deviation, Abdulhai
   consistency triplet) and claims **format-bleed rate** (H32) as its original contribution.
5. **H17 baseline expanded** — must benchmark against llama.cpp CUDA INT3 path in addition to
   BitsAndBytes NF4.

## NEW PRIORITY ORDER (systems first; empirical GPU work is the unblocking dependency)

| Priority | Item | Why |
|---|---|---|
| P0 | Physical T4 verification (NEXT_STEPS.md, unchanged) | Everything is simulation until this runs |
| P1 | H17 mega-kernel (+ folded H25, llama.cpp baseline) | Strongest unclaimed systems claim |
| P1 | H27 MXFP4-on-Turing | Rides an active 2026 wave; reuses H4/H7 machinery |
| P2 | H26 quantization×steering stability | Zero prior work; gates H19; small scope |
| P2 | H29 W1.58A4 CUDA | Completes BitNet-on-GPU gap |
| P3 | H28 + reframed H20 | Publishable only with new framing |
| P3 | H32/H33 (eval engineering) | Supports the persona product line |
| P4 | H31 (multi-persona coexistence) | Needs H26 first |
| P4 | H30 (compiler framework) | Aspirational; only after H8 is empirically verified |
