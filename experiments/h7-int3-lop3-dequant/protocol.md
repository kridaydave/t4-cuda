# Protocol for H7: INT3 Dequantization via LOP3 Bit Manipulation on Turing CC 7.5

## 1. Hypothesis
Packed 3-bit integer (INT3) weight dequantization using `LOP3.LUT` bit manipulation combined with magic floating-point exponent insertion on NVIDIA Turing SM 7.5 (Tesla T4):
1. Reduces SASS dequantization instruction count from **14.0 instructions/word** to **5.0 instructions/word** (a **2.80x SASS instruction reduction**).
2. Eliminates 32-way shared memory (SMEM) bank conflicts down to **0 bank conflicts** via 16-byte XOR swizzling.
3. Increases arithmetic intensity from **64.0 FLOP/byte** (FP16) to **341.3 FLOP/byte** (a **5.33x memory traffic reduction**).
4. Bypasses the GDDR6 memory bandwidth bottleneck on Tesla T4, achieving a predicted **2.65x roofline speedup** over standard FP16 GEMM for memory-bound prefill/decode weights.

---

## 2. Motivation
LLM inference and QLoRA fine-tuning on consumer/edge GPUs like Tesla T4 (16 GB GDDR6, 320 GB/s bandwidth) are severely limited by memory bandwidth. INT3 quantization packs 10 x 3-bit signed values into a single 32-bit register (with 2 padding bits), or 32 values across 3 x 32-bit registers (96 bits total). 

However, NVIDIA Turing CC 7.5 lacks native INT3 Tensor Core instructions. Standard PTX/SASS dequantization relies on bitfield extraction (`BFE.U32`), arithmetic shifts (`SRA`), and integer-to-float conversions (`I2F`), creating severe instruction-issue pipeline bottlenecks that degrade ALU utilization. By exploiting the Turing `LOP3.LUT` instruction—which evaluates arbitrary 3-input boolean truth tables in a single clock cycle—we can combine mantissa mask extraction, sign adjustment, and FP16 magic exponent insertion (`0x6400`) into minimal SASS cycles.

---

## 3. Mathematical Derivation

### 3.1 Bit Packing & Sub-Byte Representation
Let $x \in [-4, 3]$ be a 3-bit signed integer in 2's complement representation using bits $b_2 b_1 b_0$. Ten such integers are packed into a 32-bit word $W$:
$$W = \sum_{k=0}^{9} x_k \cdot 2^{3k}$$

### 3.2 SASS Instruction Count Comparison
**Baseline BFE Dequantization Sequence (per 32-bit register holding 2 elements being unpacked to FP16x2):**
1. `BFE.U32 R1, R_packed, 0, 3` (Extract elem 0)
2. `BFE.U32 R2, R_packed, 3, 3` (Extract elem 1)
3. `SHL R1_s, R1, 29`
4. `SRA R1_signed, R1_s, 29` (Sign extend elem 0)
5. `SHL R2_s, R2, 29`
6. `SRA R2_signed, R2_s, 29` (Sign extend elem 1)
7. `I2F.F16 R1_fp16, R1_signed` (Convert elem 0 to FP16)
8. `I2F.F16 R2_fp16, R2_signed` (Convert elem 1 to FP16)
9. `PRMT R_pair, R1_fp16, R2_fp16, 0x3210` (Pack into FP16x2)
10. `HFMA2 R_out, R_pair, R_scale, R_zero` (Scale)
+ 4 loop/index management instructions = **14 SASS instructions per 32-bit register**.

**Optimized LOP3 Magic Exponent Insertion Sequence:**
We exploit the IEEE 754 FP16 format where $1.0 \times 2^{10} + m = 1024 + m$. The FP16 exponent for $2^{10}$ is $15 + 10 = 25$ (`0x19` $\to$ bit offset 10 = `0x6400`).
1. `LOP3.LUT R_mant, R_packed, C_mask, C_magic_exp, 0xAA` (Extract mantissa and insert `0x6400` exponent in 1 cycle)
2. `LOP3.LUT R_sign, R_packed, C_sign_mask, R_zero, 0xC0` (Extract sign bit in 1 cycle)
3. `XOR R_fp16_raw, R_mant, R_sign` (Combine sign and mantissa)
4. `HSUB2 R_fp16, R_fp16_raw, C_magic_offset` (Subtract magic offset $1024.0$ in 1 cycle)
5. `HFMA2 R_out, R_fp16, R_scale, R_zero` (Scale)
Total = **5 SASS instructions per 32-bit register**.

Instruction Reduction Factor:
$$\text{Reduction} = \frac{14.0}{5.0} = 2.80\text{x}$$

### 3.3 SMEM Bank Conflict Swizzling
Shared memory on Turing has 32 banks, each 4 bytes wide. Accessing unpacked 32-bit entries linearly across 32 threads in a warp without swizzling yields 32-way bank conflicts.
We define an XOR permutation swizzle for SMEM layout:
$$\text{Bank}_{\text{swizzled}} = \left( \text{row\_idx} \oplus (\text{col\_idx} \gg 2) \right) \pmod{32}$$
Since $\text{row\_idx} \oplus (\text{col\_idx} \gg 2)$ forms a bijection over $[0, 31]$ for each warp lane, every thread accesses a unique bank, reducing bank conflicts to **0 cycles**.

### 3.4 Arithmetic Intensity & Memory Traffic
For a weight matrix of shape $M \times K = 4096 \times 4096$ ($16,777,216$ elements):
- **FP16 Baseline Traffic**: $M_{\text{FP16}} = 16,777,216 \times 2 \text{ bytes} = 33,554,432 \text{ bytes } (33.55 \text{ MB})$.
- **INT3 Packed Traffic**: $M_{\text{INT3}} = 16,777,216 \times 0.375 \text{ bytes} = 6,291,456 \text{ bytes } (6.29 \text{ MB})$.
- **FLOPs**: $2 \times M \times N \times K = 2 \times 4096 \times 4096 \times 4096 = 1.374 \times 10^{11} \text{ FLOPs}$ (assuming $N=4096$).

$$\text{AI}_{\text{FP16}} = \frac{2 \cdot N_{\text{params}}}{2.0 \text{ bytes/param}} = 64.0 \text{ FLOP/byte}$$
$$\text{AI}_{\text{INT3}} = \frac{2 \cdot N_{\text{params}}}{0.375 \text{ bytes/param}} = 341.3 \text{ FLOP/byte}$$

### 3.5 Tesla T4 Roofline Model & Speedup
Tesla T4 Specifications:
- Peak Memory Bandwidth: $B_{\text{GDDR6}} = 320 \text{ GB/s}$
- Peak FP16 Tensor Core Performance: $P_{\text{FP16\_TC}} = 65.0 \text{ TFLOPS}$
- Roofline Knee Point: $AI^* = \frac{65.0 \times 10^{12}}{320.0 \times 10^9} = 203.125 \text{ FLOP/byte}$

FP16 GEMM is memory-bound ($64.0 < 203.125$), capping performance at:
$$P_{\text{FP16}} = 64.0 \text{ FLOP/byte} \times 320 \text{ GB/s} = 20.48 \text{ TFLOPS}$$

INT3 GEMM has $\text{AI}_{\text{INT3}} = 341.3 > 203.125$, transitioning the kernel into the compute-bound regime. Bounded by LOP3 dequantization execution efficiency ($\eta_{\text{dequant}} \approx 0.83$):
$$P_{\text{INT3}} = \min\left(65.0, 341.3 \times 0.320 \times 0.83\right) \approx 54.3 \text{ TFLOPS}$$
$$\text{Speedup} = \frac{54.3 \text{ TFLOPS}}{20.48 \text{ TFLOPS}} = 2.65\text{x}$$

---

## 4. Execution Protocol
1. Implement `research/src/simulate_h7_int3_lop3.py` to model SASS instruction parsing, bank conflict matrices, arithmetic intensity, memory traffic, and roofline curves.
2. Simulate standard BFE vs LOP3 dequantization SASS sequences across 10,000 synthetic 32-bit packed registers.
3. Compute 32-thread SMEM access patterns with and without 16-byte XOR swizzling.
4. Calculate exact DRAM bytes transferred and roofline throughput on Tesla T4.
5. Export structured results to `research/experiments/h7-int3-lop3-dequant/analysis.md`.

---

## 5. Predictions
- SASS Instruction Count: 14 instrs (BFE) vs 5 instrs (LOP3) $\implies$ **2.80x reduction**.
- SMEM Bank Conflicts: 32-way (Linear) vs 0 (XOR Swizzle) $\implies$ **100% elimination**.
- Memory Traffic Reduction: 33.55 MB (FP16) vs 6.29 MB (INT3) $\implies$ **5.33x reduction**.
- Arithmetic Intensity: 64.0 FLOP/byte $\to$ **341.3 FLOP/byte**.
- Tesla T4 Roofline Speedup: **2.65x speedup**.

---

## 6. Analysis Plan
- Verify exact instruction counts against Turing ISA `LOP3.LUT` truth tables (e.g., LUT `0xAA`, `0xC0`).
- Compare memory traffic metrics against baseline FP16 and INT8 quantized representations.
- Plot/tabulate Roofline knee point transition on T4 architecture.
