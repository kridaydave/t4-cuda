# Hypothesis 4 (H4)

**Hypothesis:** Signed INT4 Two's Complement dequantization via LOP3 LUT 0x6A reduces instruction counts by 2.5x compared to standard BFE (Bit-Field Extract) sequence on Turing SM 7.5.

**Protocol:**
1. Formulate the instruction sequence for standard INT4 dequantization using BFE instructions.
2. Formulate the instruction sequence for INT4 dequantization using LOP3 with LUT 0x6A.
3. Compare the instruction counts per 32-bit register (which holds 8 packed INT4 values).
4. Check if the reduction is 2.5x.
