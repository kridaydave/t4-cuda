# Tesla T4 (Turing CC 7.5 / TU104) PTX Assembly & SASS Deep Dive

This report provides an ultra-deep dive into Turing's low-level architecture, exploring PTX inline assembly, SASS disassembly, execution control codes, and Turing's uniform datapath.

## 1. PTX Inline Assembly & SASS Disassembly Analysis

### LOP3 Truth Tables
The `LOP3` instruction in PTX and SASS allows performing arbitrary 3-input bitwise logical operations in a single cycle. It takes three inputs and an 8-bit immediate truth table (LUT).

- **0x78**: Often used for signed combinations or conditionally flipping bits based on a mask. Truth table `0x78` corresponds to `(A ^ B) & C` or similar variations.
- **0x64**: Used for specific bit-packing and exponent injections in custom data types.
- **0xE2**: `(A & B) | (A & C) | (B & C)` - Majority gate.
- **0xF2**: `(A | B) | (A & ~C)` - Custom logical combination.

**Inline PTX Example:**
```cpp
// LOP3 block
asm volatile("lop3.b32 %0, %1, %2, %3, 0x78;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
```

### Turing 128-bit SASS Instruction Word Control Codes
On Turing (TU104), SASS instructions are 128 bits wide. Unlike previous architectures where scheduling was purely hardware-driven or used separate control instructions, Turing embeds control codes directly into each instruction word.
- **Stall Cycles (Wait State Count):** Encodes how many cycles the warp scheduler should wait before issuing the next instruction.
- **Yield Flag:** Indicates if the warp can yield execution to another warp.
- **Read/Write Barrier Masks:** Enforces execution order by setting and waiting on barrier registers (usually 6 barrier registers per warp) to prevent read-after-write (RAW), write-after-write (WAW), and write-after-read (WAR) hazards.

### Uniform Registers and Uniform Datapath
Turing introduced a dedicated Uniform Datapath with Uniform Registers (UR0-UR63).
- **Purpose:** Offload loop invariants, constant offsets, and pointer arithmetic from the general-purpose registers (GPRs).
- **Instructions:** `UNOP`, `UIADD3`, `ULEA`.
- **Benefit:** Reduces GPR pressure and saves datapath power, as uniform instructions are executed once per warp rather than per thread.

## 2. Low-Level PTX Matrix Primitives & Warp Crossbar Shuffles

### Tensor Core Primitives: LDMATRIX
`ldmatrix` loads data from shared memory directly into the registers required for Tensor Core operations.
- **Format:** `ldmatrix.sync.aligned.m8n8.x4.shared.b64` (Load four 8x8 matrices)
- **Format (Transposed):** `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b64`

### Tensor Core Primitives: MMA
The `mma` instruction performs matrix multiply-accumulate.
- **Format:** `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`
- Multiplies an A matrix (FP16, row-major) by a B matrix (FP16, col-major) and accumulates into a C matrix (FP32).

### Warp Shuffle Crossbar Instructions
Warp shuffles use the crossbar network to exchange data directly between threads without using shared memory.
- `shfl.sync.idx.b32`: Fetch data from a specific thread ID in the warp.
- `shfl.sync.bfly.b32`: Butterfly shuffle (XOR thread ID), highly useful for parallel reductions (e.g., butterfly sum reduction).
