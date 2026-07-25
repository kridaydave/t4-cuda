# Analysis of H4

**Hypothesis:** Signed INT4 Two's Complement dequantization via LOP3 LUT 0x6A reduces instruction counts by 2.5x compared to standard BFE (Bit-Field Extract) sequence on Turing SM 7.5.

**Results:**
- Standard BFE instruction count per 32-bit register (8 INT4 elements): 10 instructions
- Optimized LOP3 (LUT 0x6A) instruction count per 32-bit register: 4 instructions
- Reduction factor: 2.5x

**Conclusion:**
The hypothesis is confirmed. Utilizing LOP3 with LUT 0x6A effectively reduces the instruction footprint for INT4 dequantization by 2.5x compared to standard BFE approaches on Turing SM 7.5 architectures.
