# CRITICAL AUDIT: Hardware, PTX, and Assembly Research on Tesla T4 (Turing CC 7.5)

## Executive Summary
This document provides a rigorous technical audit of the low-level hardware and PTX research for the Tesla T4 (TU104). It identifies edge-case risks, hidden assumptions, and microarchitectural boundary failures in the proposed execution pipelines. For each identified issue, an explicit remediation strategy is provided.

---

## 1. Register Pressure & Local Memory Spills

**Technical Issue / Risk:**
The proposed 25% occupancy model relies heavily on capping registers at 64 per thread (`--maxrregcount=64`) to prevent local memory spilling. However, if complex math or aggressive unrolling forces the compiler to exceed 64 registers, `ptxas` will spill to Local Memory. On Turing, Local Memory resides in GDDR6 DRAM (cached in L1/L2). Because the pipeline relies on strict deterministic latency hiding (e.g., 512 math cycles covering 450 memory cycles), any unexpected Local Memory spill injects hundreds of stall cycles, completely breaking the Little's Law latency-hiding math and plummeting TFLOPS.

**Explicit Mitigation Strategy:**
1. **Algorithmic Cap:** Restrict pipeline depth strictly to **2-stage double buffering** (which empirical tests show uses ~52 registers/thread) rather than 3-stage or 4-stage buffering.
2. **Uniform Datapath Offloading:** Leverage Turing's specific **Uniform Registers (UR0-UR63)** and uniform instructions (`UIADD3`, `ULEA`) for loop invariants, array strides, and pointer arithmetic. This offloads GPR pressure to the uniform datapath, keeping thread-specific GPR usage safely below the 64-register limit.
3. **Compiler Verification:** Mandate `-Xptxas -v` in the build system and parse the output. If `bytes spill stores` > 0, the build must fail.

---

## 2. Tile Under-Utilization & Wave-Tail Imbalance

**Technical Issue / Risk:**
The "Persistent Grid Block Streaming" strategy mandates launching exactly 40 thread blocks (1 per SM). While this is highly effective for large matrices (e.g., $M=4096, N=4096$), a critical hidden assumption is that the total number of macro-tiles ($M/128 \times N/128$) is greater than or equal to 40. For small shapes (e.g., $M=64, N=64$, typical in early layers or autoregressive decoding), there is only 1 macro-tile. Launching 40 persistent blocks will leave 39 SMs completely idle, resulting in severe hardware under-utilization (2.5% effective SM utilization).

**Explicit Mitigation Strategy:**
1. **Dynamic Grid Dispatch Fallback:** Implement a dynamic heuristic in the host launch code. If `(M / TILE_M) * (N / TILE_N) < 40`, ABANDON the 40-block persistent grid strategy.
2. **Micro-Tile Specialization:** For small matrices, switch to a smaller tile size (e.g., $64 \times 64$ or $32 \times 32$) to artificially increase the tile count and spread work across all 40 SMs.
3. **Batched GEMM / Split-K:** For thin matrices (e.g., $M=1$ or $M=16$), use Split-K partitioning across the reduction dimension to ensure all 40 SMs are fed with partial reduction tasks.

---

## 3. Execution Port Contention & Dual-Issue Stalls

**Technical Issue / Risk:**
Turing's warp scheduler cannot dual-issue instructions from the same warp in a single clock cycle. The research models compute throughput based on 128 `mma.sync` instructions, but fails to account for the heavy mix of INT32 address calculation, `LOP3.LUT` bitwise logic, and FP32 accumulation happening concurrently. If a single warp interleaves memory pointer arithmetic (INT32), accumulator updates (FP32), and Tensor Core math (HMMA), the warp scheduler will face severe instruction port contention, introducing structural stalls.

**Explicit Mitigation Strategy:**
1. **Warp-Specialized Producer-Consumer Pipeline:** Strictly decouple the instruction streams. Dedicate 2 warps entirely to GMEM/SMEM memory movement and address calculation (Producer), and 6 warps entirely to `mma.sync` math and FP32 accumulation (Consumer). By placing them on different sub-cores, the SM can independently issue INT32/LSU instructions on Sub-Core 0, while Sub-Cores 1-3 continuously issue Tensor/FP32 instructions without intra-warp contention.
2. **Explicit SASS Yield Flags:** Use PTX inline assembly to carefully manage control codes, ensuring math warps do not yield unexpectedly while holding critical register operands.

---

## 4. NVPM Power Throttling Transient Spikes

**Technical Issue / Risk:**
While capping occupancy at 25% ensures sustained power remains around ~62W (under the 70W TDP), the research assumes a perfectly flat power profile. However, during a "cold-start" of the 40 persistent blocks, the simultaneous global fetch (`LDG.E.128`) from all 40 SMs into the L2 cache, followed by an immediate simultaneous burst of Tensor Core `mma.sync`, can create an instantaneous sub-millisecond power transient spike. The NVPM PID controller might detect this dI/dt (current spike) and conservatively drop the clock from 1590 MHz to 1100 MHz for a few milliseconds before recovering, causing micro-jitters in execution time.

**Explicit Mitigation Strategy:**
1. **Atomic Tile Fetch Staggering:** Modify the persistent block worker loop so that Warp 0's atomic fetch from the global L2 counter is slightly staggered. Introduce a lightweight backoff or offset based on `blockIdx.x` (the SM ID).
2. **Warp Staggered Initialization:** Phase the start of the Producer warps. Let SMs 0-19 fetch their first tile, wait a few cycles, then let SMs 20-39 fetch. This flattens the initial L2 bandwidth surge and smooths the dI/dt power curve, ensuring NVPM never detects a transient violation.
