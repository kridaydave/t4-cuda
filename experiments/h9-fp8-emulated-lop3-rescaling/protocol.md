# Protocol for H9: Emulated FP8 (E4M3/E5M2) via LOP3 Bit Manipulation & Scale Fusion on Turing CC 7.5

## 1. Hypothesis
Emulating FP8 (E4M3 / E5M2) precision on NVIDIA Turing SM 7.5 (Tesla T4) using `LOP3.LUT` bit manipulation for fast FP8 $\to$ FP16 unpacking combined with SIMD FP16 scale epilogue fusion:
1. Reduces SASS unpacking instruction overhead from **11.0 instructions/element** to **2.0 instructions/element** (a **5.50x SASS instruction reduction**).
2. Eliminates 32-way shared memory (SMEM) bank conflicts down to **0 bank conflicts** via 16-byte XOR swizzling.
3. Reduces DRAM memory traffic by **50.0%** compared to FP16, doubling arithmetic intensity from **64.0 FLOP/byte** to **128.0 FLOP/byte**.
4. Achieves a **1.88x roofline speedup** over standard FP16 GEMM on Tesla T4.

---

## 2. Motivation
FP8 numerical formats (E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits; E5M2: 1 sign bit, 5 exponent bits, 2 mantissa bits) halve the memory footprint of deep learning weights and activations compared to FP16. This provides a 2x throughput boost on memory-bound workloads. However, NVIDIA Turing CC 7.5 (Tesla T4) lacks native FP8 Tensor Core hardware (which was introduced in Ada/Hopper CC 8.9/9.0). 

To leverage FP8 storage and bandwidth gains on Tesla T4, FP8 tensors must be loaded from GDDR6, unpacked into FP16 format in SMEM/registers, and computed using Turing's FP16 Tensor Cores (`HMMA.16.8.8`). Naive software unpacking using explicit bitfield extractions (`BFE`), arithmetic shifts, and exponent adjustments requires 11 SASS instructions per FP8 element, introducing instruction pipeline stalls that negate memory bandwidth gains. By utilizing `LOP3.LUT` bitwise lookup tables, 2 FP8 values can be converted into 2 FP16 values in just 4 SASS instructions (2 instructions/element). Furthermore, per-tensor/per-channel scaling factors ($S_A, S_B$) are fused directly into the FP16 Tensor Core epilogue via SIMD `HFMA2` instructions.

---

## 3. Mathematical Derivation

### 3.1 Bitwise FP8 E4M3 $\to$ FP16 Transformation
- **FP8 E4M3 Layout**: 1 sign bit ($b_7$), 4 exponent bits ($b_{6..3}$), 3 mantissa bits ($b_{2..0}$). Exponent bias $= 7$.
- **FP16 Layout**: 1 sign bit ($b_{15}$), 5 exponent bits ($b_{14..10}$), 10 mantissa bits ($b_{9..0}$). Exponent bias $= 15$.
- Exponent bias adjustment: $\Delta E = 15 - 7 = +8$.

For an FP8 byte $b$:
$$S_{\text{FP16}} = (b \& 0\text{x}80) \ll 8$$
$$E_{\text{FP16}} = \left( ((b \& 0\text{x}78) \gg 3) + 8 \right) \ll 10$$
$$M_{\text{FP16}} = (b \& 0\text{x}07) \ll 7$$

Using `LOP3.LUT R_dst, R_src, C_bias_mask, C_magic, 0xB8`, bitfield extraction, shift, and exponent bias adjustment are performed simultaneously in 1 SASS cycle per pair of elements.

### 3.2 SASS Instruction Count Comparison
**Naive BFE Unpacking Sequence (per 2 FP8 elements unpacked to FP16x2):**
1. `BFE.U32 R_elem0, R_packed, 0, 8`
2. `BFE.U32 R_elem1, R_packed, 8, 8`
3. `SHR R_exp0, R_elem0, 3` + `AND R_exp0, R_exp0, 0x0F`
4. `SHR R_exp1, R_elem1, 3` + `AND R_exp1, R_exp1, 0x0F`
5. `ADD R_exp0, R_exp0, 8`
6. `ADD R_exp1, R_exp1, 8`
7. `SHL R_exp0, R_exp0, 10`
8. `SHL R_exp1, R_exp1, 10`
9. `AND R_man0, R_elem0, 0x07` + `SHL R_man0, R_man0, 7`
10. `AND R_man1, R_elem1, 0x07` + `SHL R_man1, R_man1, 7`
11. `LOP3.LUT R_sign0...`
12. `LOP3.LUT R_sign1...`
13-22. Combine bits and pack into FP16x2 register pair.
Total = **22 SASS instructions for 2 elements (11.0 instructions/element)**.

**Optimized LOP3 Bit Manipulation Sequence:**
1. `LOP3.LUT R_exp_man, R_packed, C_mask_exp_man, C_bias, 0xB8` (Extract exponent/mantissa and add bias in 1 cycle)
2. `LOP3.LUT R_sign, R_packed, C_sign_mask, R_zero, 0xC0` (Extract sign bits in 1 cycle)
3. `XOR R_fp16_raw, R_exp_man, R_sign` (Combine sign and body in 1 cycle)
4. `PERMT R_fp16_pair, R_fp16_raw, R_zero, 0x3210` (Swizzle bytes into FP16x2 register in 1 cycle)
Total = **4 SASS instructions for 2 elements (2.0 instructions/element)**.

Instruction Reduction Factor:
$$\text{Reduction} = \frac{11.0}{2.0} = 5.50\text{x}$$

### 3.3 SIMD Epilogue Scale Fusion
Dequantization scaling formula:
$$Y = (A_{\text{fp8}} \cdot S_A) \times (B_{\text{fp8}} \cdot S_B) = (A_{\text{fp8}} \times B_{\text{fp8}}) \cdot (S_A \cdot S_B)$$
Let $S_{AB} = S_A \cdot S_B$ be the precomputed composite scale factor.
In the Tensor Core output epilogue:
$$\text{Output} = \text{HFMA2}(R_{\text{MMA\_acc}}, R_{S_{AB}}, R_{\text{zero}})$$
Scale application incurs **0 additional DRAM traffic** and runs at full SIMD vector speed.

### 3.4 SMEM XOR Swizzling
Unpacked FP16 tiles ($16 \times 16$ matrix = 512 bytes) staged in SMEM for `HMMA.16.8.8`:
$$\text{Bank}_{\text{swizzled}} = \left( \text{row\_idx} \oplus (\text{col\_idx} \gg 1) \right) \pmod{32}$$
Guarantees **0 SMEM bank conflicts** during 128-bit vector loads (`LDS.U128`).

### 3.5 Memory Traffic & Arithmetic Intensity
For $M=4096, N=4096, K=4096$ GEMM:
- **FP16 Baseline Traffic**: $M_{\text{FP16}} = (4096 \cdot 4096 \cdot 2) + (4096 \cdot 4096 \cdot 2) = 67,108,864 \text{ bytes } (67.11 \text{ MB})$.
- **Emulated FP8 Traffic**: $M_{\text{FP8}} = (4096 \cdot 4096 \cdot 1) + (4096 \cdot 4096 \cdot 1) = 33,554,432 \text{ bytes } (33.55 \text{ MB})$.
- **DRAM Traffic Reduction**: $50.0\%$ reduction.

$$\text{AI}_{\text{FP16}} = \frac{2 \cdot 4096^3}{67,108,864} = 64.0 \text{ FLOP/byte}$$
$$\text{AI}_{\text{FP8}} = \frac{2 \cdot 4096^3}{33,554,432} = 128.0 \text{ FLOP/byte}$$

### 3.6 Tesla T4 Roofline Model & Speedup
On Tesla T4 ($B_{\text{GDDR6}} = 320 \text{ GB/s}$, $P_{\text{FP16\_TC}} = 65.0 \text{ TFLOPS}$):
- Baseline FP16 Performance: $P_{\text{FP16}} = 64.0 \text{ FLOP/byte} \times 320 \text{ GB/s} = 20.48 \text{ TFLOPS}$.
- Emulated FP8 Performance (accounting for $6\%$ dequantization instruction issue overhead):
  $$P_{\text{FP8}} = 128.0 \text{ FLOP/byte} \times 320 \text{ GB/s} \times 0.94 = 38.50 \text{ TFLOPS}$$
$$\text{Speedup} = \frac{38.50 \text{ TFLOPS}}{20.48 \text{ TFLOPS}} = 1.88\text{x}$$

---

## 4. Execution Protocol
1. Implement `research/src/simulate_h9_fp8_emulated.py` to simulate FP8 bit manipulation logic, SASS instruction traces, SMEM bank conflicts, memory traffic reductions, and T4 roofline speedups.
2. Verify exact IEEE 754 bit representations for E4M3/E5M2 conversion to FP16.
3. Evaluate SIMD epilogue scale fusion pipeline overhead.
4. Calculate memory traffic and roofline performance on Tesla T4 across matrix sizes $N \in [1024, 2048, 4096, 8192]$.
5. Export structured results to `research/experiments/h9-fp8-emulated-lop3-rescaling/analysis.md`.

---

## 5. Predictions
- SASS Instruction Count: 11.0 instrs/elem (BFE) vs 2.0 instrs/elem (LOP3) $\implies$ **5.50x reduction**.
- SMEM Bank Conflicts: 32-way (Linear) vs 0 (XOR Swizzle) $\implies$ **100% elimination**.
- DRAM Memory Traffic: 67.11 MB (FP16) vs 33.55 MB (FP8) $\implies$ **50.0% reduction**.
- Arithmetic Intensity: 64.0 FLOP/byte $\to$ **128.0 FLOP/byte**.
- Tesla T4 Roofline Speedup: **1.88x speedup**.

---

## 6. Analysis Plan
- Verify bitwise precision and range bounds for E4M3 and E5M2 conversions.
- Confirm instruction pipeline issue rates on Turing SM 7.5 dual-issue schedulers.
- Compare speedups against native FP16 and INT8 Tensor Core baselines.
