# Literature: 2025-2026 Research Landscape Survey
## Autoresearch Outer Loop — 2026-07-26

### Sources Searched
- Web search (3 parallel research scouts)
- ArXiv 2025-2026
- Open-source repositories (GitHub)
- Conference proceedings (NeurIPS, ICML, ASPLOS, MLSys)

---

## 1. Sub-Byte Quantization (2025-2026)

### ARCQuant & ScaleSweep (2025-2026)
- **Focus:** FP4 (NVFP4) with block-wise scaling and differentiable estimators during training
- **T4 Relevance:** Since T4 lacks native FP4 instructions, backport the LUT approach — pack FP4 into FP16 via LOP3 exponent tricks

### BitsMoE (2025)
- **Focus:** Signed INT3 with absmax scaling, dynamically allocating bits based on block sensitivity
- **T4 Relevance:** Directly validates our H7 INT3 work; consider adopting dynamic bit allocation per block

### DeepGEMM / PolyQ (2025-2026)
- **Focus:** LUT-based sub-byte kernels for non-native hardware (CPUs, pre-Blackwell GPUs)
- **T4 Relevance:** Architecturally similar to our LOP3 approach — worth comparing instruction counts

---

## 2. Personality & Activation Steering (2025-2026)

### Representation Engineering (RepE) — Extended (2026)
- **Core concept:** Treat personality/behavior as "directions" in LLM latent space
- **Persona Algebra:** Frameworks like PERSONA and SAS allow adding, subtracting, scaling multiple trait vectors
- **Situational Steering:** IRIS (Identify-Retrieve-Steer) and Dynamic Persona Routing (DPR) adjust steering based on conversation context

### GCAD — Gated Cropped Attention-Delta (2026)
- **Breakthrough:** Solves "coherence collapse" in long contexts by steering at the attention level rather than residual stream
- **Prevents KV-cache contamination**
- **Critical for Epoch:** Multi-turn persona persistence without degradation

### Open-Source Implementations
- `IBM/activation-steering` — Conditional Activation Steering (CAST) for dynamic, context-aware control
- `cma1114/activation_steering` — Research-oriented, flexible hooking at arbitrary layers
- `annahdo/implementing_activation_steering` — Minimal PyTorch hooks + TransformerLens
- `microsoft/llm-steer-instruct` — Steering for instruction-following enhancement

### Key Technical Details
1. **Extraction:** Run paired contrastive prompts through model. Average activation difference = persona vector
2. **Layer Selection:** Middle-to-late layers optimal for persona control
3. **Application:** persona_vector × α added to activations during generation
4. **Comparison:** Steering > System Prompts (no drift, no token cost) and > LoRA (no training needed, dynamic per-request)

---

## 3. Code Quality / Taste Evaluation (2025-2026)

### Senior SWE-Bench (2026)
- **New benchmark:** Evaluates "taste" — code reflecting the architectural judgment of a senior engineer
- **Current SOTA:** Top frontier models (Claude Opus 4.8) achieve only ~24% "tasteful solve" rate
- **Significance:** Taste is a genuine frontier problem, not solved

### LLM-as-a-Judge for Code Quality
- **Pairwise comparison** is significantly more reliable than scalar scoring
- **Chain-of-Thought prompting** improves reliability further
- **Cross-language generalization:** Models trained via RLAIF to review Python generalize to Java

### Software Constitutions (CAI for Code)
- Replace black-box RLHF with transparent rules (PEP-8, OWASP)
- Stage 1: Model self-critiques against constitution, revises
- Stage 2: RLAIF builds preference dataset from best-aligned outputs
- Result: "Security by construction"

### OpenDesign Benchmark (2025-2026)
- Focuses on aesthetic and structural quality of UI/frontend code
- Utilizes multi-agent aesthetics critiquing
- Relevant to Epoch's frontend taste training

---

## 4. Fused Kernel Innovation (2025-2026)

### Lazy Pre-Norm & Multi-CTA Norm Fusion
- Fuse RMSNorm directly into backward GEMM or attention epilogues
- "Fusion regrouping" shifts normalizations across layers to maintain fusion viability
- Extends our H6 (Fused Backward AdamW)

### Kernel-Smith (2025)
- RL-based auto-generation of verified CUDA kernels
- Uses actual hardware profiling (Nsight Compute latency) as reward signal
- Could close the loop: Albert generates PTX → profile → reward → better PTX

### Marlin Kernel Architecture Details
- Asynchronous Data Movement via `cp.async` (Ampere+)
- L2 cache + SMEM pipelining with circular buffers
- "On-the-fly" dequantization entirely within registers
- Tensor Core saturation via large contiguous block loads (128 bytes/warp)
- **Limitation:** Requires `cp.async` — our H17 must replicate this via software warp specialization

### BitBLAS Framework
- Compiler-based (TIR Script) for custom mixed-precision matrix operations
- Generates hardware-aware CUDA kernels for non-standard formats (INT4, INT2, NF4)
- Integrates with PyTorch, vLLM, AutoGPTQ
- Could serve as alternative to hand-written CUDA for some kernels

---

## 5. Framework Updates (2025-2026)

### ThunderKittens 2.0 (Jan 2026)
- **Dropped Ampere/Turing support** — Hopper/Blackwell only with NVFP4/MXFP8
- Register layout logic can still be manually backported for our use

### FlashInfer (2025-2026)
- FP4 support, extreme optimizations for DeepSeek-R1 and Qwen3
- Best reference for low-latency serving architectures
- Older branches usable for T4-specific attention kernels

### FlexAttention FA4 (2026)
- Integrated with FlashAttention-4
- Custom attention masks (sliding window, ALiBi) written in PyTorch, compiled via TorchInductor
- Can prototype Epoch models' attention patterns before writing raw CUDA

---

## 6. Speculative Decoding (2025-2026)

### Quantized Draft Models
- INT8: Safe with minimal accuracy degradation
- INT4: Risks lowering acceptance rate due to quantization noise
- AWQ/GPTQ post-training quantization required to protect outlier weights

### T4 Characteristics
- T4's memory-bandwidth bottleneck (320 GB/s) makes speculative decoding especially effective at batch_size=1
- Must keep draft model overhead minimal to avoid negating benefits
- As batch size increases, baseline throughput improves and speculative speedup diminishes

### Optimal Draft Model Sizing
- **0.5-1.5B draft for 32B target** (consensus)
- 4B draft for 32B target is SUBOPTIMAL — too large
- Both MUST share exact tokenizer vocabulary
- Alternatives: Medusa, EAGLE-3 (auxiliary heads, no separate draft model)
