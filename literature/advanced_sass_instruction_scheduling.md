# Advanced SASS Instruction Scheduling for Turing SM 7.5

## Abstract

This paper explores advanced SASS (Streaming Assembler) instruction scheduling on the NVIDIA Turing SM 7.5 microarchitecture. We specifically focus on optimizing instruction latencies, maximizing dual-issue pipeline utilization, eliminating shared memory bank conflicts via XOR swizzling, and formatting register fragments for `HMMA.884` Tensor Core instructions. These optimizations are evaluated in the context of custom INT4 dequantization via `LOP3` / `PRMT` and mixed-precision matrix multiplication.

## 1. Turing SM 7.5 Pipeline & Dual-Issue Opportunities

The Turing SM 7.5 introduces independent integer and floating-point data paths, enabling concurrent execution of instructions. 
Turing partitioned its SM into four processing blocks, each with its own warp scheduler and dispatch unit. 

### 1.1 Instruction Latencies
Key SASS instructions involved in dequantization and MMA exhibit the following typical pipeline latencies (in cycles):
*   **FFMA / FADD / FMUL**: ~4 cycles.
*   **IADD / IMAD / LOP3 / PRMT**: ~4-6 cycles.
*   **HMMA.884 / HMMA.1688**: Fixed throughput, often taking ~14-16 cycles to fully retire dependencies for the accumulator matrix (register C).
*   **LDS / STS (Shared Memory)**: ~20-30 cycles (uncontended).

### 1.2 Dual-Issue Slots
To achieve peak performance, instruction scheduling must leverage dual-issue. SASS scheduling control codes allow encoding stall counts and yield/read/write barrier flags. On SM 7.5, integer operations (e.g., `LOP3` for bitwise operations, `PRMT` for byte manipulation, `IADD` for pointer arithmetic) can be co-issued with floating-point operations (`FFMA`) or Tensor Core instructions (`HMMA`). 

For INT4 dequantization:
*   Packed INT4 types typically arrive as 32-bit registers (8x INT4 values).
*   Extracting these values requires a combination of `LOP3` (to mask bits), `SHF` (shift), and `PRMT` (byte permutation).
*   By interleaving `HMMA` accumulation with `LOP3` dequantization of the *next* block of data, warp schedulers can mask the latency of the arithmetic operations.

## 2. Shared Memory Bank Conflict Elimination via XOR Swizzling

Turing shared memory operates across 32 banks (4 bytes per bank). When multiple threads in a warp access the same bank but different addresses, a bank conflict occurs, serializing memory accesses. 

### 2.1 The Bank Conflict Problem in Matrix Multiplication
Loading blocks of matrix A (e.g., 16x16 or 16x8 fragments) and matrix B from shared memory to registers for HMMA often induces stride-based bank conflicts if stored in row-major or column-major format directly. 

### 2.2 XOR Swizzling Pattern
To eliminate bank conflicts, we apply a bitwise XOR mapping from the logical matrix coordinates to the physical shared memory addresses. 

Given a logical 2D index `(row, col)`, the standard 1D address is:
`addr = (row * stride) + col`

With XOR swizzling (commonly using a 128-byte phase):
`swizzled_addr = addr ^ ((row % 8) * swizzle_factor)`

In SASS, this can be efficiently computed in the address generation phase:
1. Shift the row index: `SHR.U32 R_shift, R_row, 3;` 
2. XOR with the column offset: `LOP3.LUT R_addr, R_base, R_shift, R_col, 0x96;` (where 0x96 is the LUT for XOR/ADD combinations).

When loading INT4 values, `LDS.U128` (load 128-bit) provides the highest bandwidth. Swizzling ensures that the 32 threads in a warp request addresses uniformly distributed across the 32 banks.

## 3. Tensor Core HMMA.884 and Register Fragment Packing

Turing introduces integer and sub-byte Tensor Core instructions. For mixed-precision or lower-precision ML workloads, `HMMA.884` (Half-Precision Matrix Multiply Accumulate) and integer variants (`IMMA`) are critical.

### 3.1 Register Layout
An `HMMA.1688.F16` instruction computes D = A * B + C, where:
*   **A**: 16x8 matrix. Each thread provides 4 FP16 elements (2x 32-bit registers).
*   **B**: 8x8 matrix. Each thread provides 2 FP16 elements (1x 32-bit register).
*   **C/D**: 16x8 accumulator. Each thread holds 4 elements (4x 32-bit registers for FP32 accumulation, or 2x 32-bit for FP16).

### 3.2 Packing INT4 to FP16/INT8
If implementing a custom INT4 dequantization pipeline targeting FP16 Tensor Cores:
1. **Load**: `LDS.U128` loads 32 bytes (64 INT4 weights) per thread.
2. **Dequantize**: Use `PRMT` and `LOP3` to expand INT4 nibbles into 16-bit or 32-bit formats. 
3. **Format Conversion**: Convert expanded integers to FP16 (`I2F.F16.S32`).
4. **Pack**: Pack two FP16 values into a single 32-bit register (`PRMT` or `BFI`) to match the `HMMA.884` or `HMMA.1688` register layout requirements.

### 3.3 Register Dependency Stalls
`HMMA` instructions have deep pipelines. The SASS scheduler must encode appropriate wait barriers (using `.W` flags in control codes) before reading the accumulator registers (matrix D) in subsequent operations.
*   **Read-after-Write (RAW)**: A minimum of ~14 cycles must elapse before the result of `HMMA` can be used. 
*   **Double Buffering**: To avoid stalling, maintain at least two sets of accumulator registers and load/compute registers. While `HMMA` computes on register set 0, the ALU can dequantize INT4 data into register set 1. 

## 4. Conclusion
Optimal utilization of Turing SM 7.5 for INT4 workloads demands precise SASS-level control. By interleaving `LOP3`/`PRMT` with `HMMA` instructions, applying XOR swizzling to shared memory offsets, and strictly managing register dependencies via double-buffering, compute bounds can approach theoretical maximums.
