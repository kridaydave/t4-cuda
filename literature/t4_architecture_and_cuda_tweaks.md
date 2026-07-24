# Tesla T4 (Turing CC 7.5) Literature & Architectural Deep-Dive

## Summary of Reference Materials & Microarchitectural Specifications

### 1. Hardware Limits & Roofline Profile
- **NVIDIA Turing Architecture (TU104)**:
  - 40 SMs, 64 FP32 Cores + 64 INT32 Cores per SM (independent execution units allowing simultaneous FP32 + INT32 instructions).
  - 8 Tensor Cores per SM = 320 Tensor Cores.
  - Memory Subsystem: 16 GB GDDR6, 256-bit interface, 10 Gbps -> **320 GB/s Peak Memory Bandwidth**.
  - L2 Cache: 4 MB.
  - Thermal / Power Cap: **70 Watt TDP**.

### 2. Microarchitectural Constraints & Latencies on T4
- **Global Memory Access Latency**: ~400–600 GPU clock cycles.
- **Shared Memory Access Latency**: ~20–30 cycles.
- **Register File Access Latency**: ~1 cycle.
- **Memory Coalescing Requirement**: GDDR6 access transactions occur in 32-byte or 64-byte bursts per memory controller. Warp accesses (32 threads) should load contiguous 128-bit vectors (`float4` / `uint4`) to achieve full 100% memory bus efficiency.
- **Bank Conflicts in Shared Memory**: 32 banks, 4 bytes wide. Loading 16x16 FP16 tiles for WMMA/Tensor Cores requires row/col stride swizzling (`smem[i ^ (j >> 2)]`) or padding (`float smem[16][18]`) to avoid 4-way to 8-way bank serialization.

### 3. Tensor Core Instructions for Turing (CC 7.5)
- **WMMA API (`nvcuda::wmma`)**:
  - Shapes: `matrix_a<half, 16, 16, 16, wmma::matrix_a, wmma::row_major>`, `matrix_b<half, 16, 16, 16, wmma::matrix_b, wmma::col_major>`, accumulator `<float, 16, 16, 16>`.
- **PTX Assembly Inline Target**:
  - `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 %0, %1, %2, %3;`
  - High performance: 16x8x8 tile calculation per warp per instruction cycle.

### 4. Software Prefetching vs Ampere `cp.async`
- Ampere (A100, RTX 3090, CC 8.0+) introduced `cp.async.ca.shared.global` which bypasses registers when copying GMEM -> SMEM.
- Turing (T4, RTX 2080, CC 7.5) DOES NOT have hardware `cp.async`.
- **Optimization Strategy**: Double buffering via Register Prefetching.
  - Stage 0: Load GMEM Tile 0 -> Regs -> SMEM Tile 0.
  - Loop step K:
    - Load GMEM Tile K+1 -> Regs_prefetch
    - Compute Tensor Core WMMA on SMEM Tile K
    - Store Regs_prefetch -> SMEM Tile K+1
    - __syncthreads()

### 5. 70W Power Budget & Occupancy Tuning
- Running 100% thread occupancy (e.g. 1024 threads/SM across all 40 SMs with heavy Tensor Core instructions) causes the T4 hardware power limiter to downclock GPU boost frequency from ~1590 MHz down to ~900–1100 MHz to respect the 70W power limit!
- **T4 Tweak**: Launching with 50%–75% occupancy (e.g., 256–512 threads per block, 1–2 blocks per SM) often yields *higher sustained clock speeds* and overall faster wall-clock execution time than 100% occupancy.
