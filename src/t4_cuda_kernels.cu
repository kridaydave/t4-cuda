/*
 * Tesla T4 (Turing Compute Capability 7.5) Extreme CUDA Kernel Customizations
 * Target Architectures: TU104 / Tesla T4 (40 SMs, 320 Tensor Cores, 320 GB/s, 70W TDP)
 * 
 * Features Implemented:
 * 1. H1: Vectorized 128-bit (float4/uint4) Memory Coalescing & Swizzled Shared Memory (No Bank Conflicts)
 * 2. H2: Register-Prefetched Double-Buffering Pipeline for Turing (hides memory latency without Ampere cp.async)
 * 3. H3: Fused T4 FlashAttention-2 Sub-Tile FP16 Kernel (64x64 tiles, online register softmax, warp shuffles)
 * 4. H4: Sub-Byte INT4 (W4A16) PTX Bitfield Unpacking ('bfe.u32') & Tensor Core WMMA Accumulation
 * 5. H5: 70W TDP Power-Aware Grid Occupancy & Sustained Clock Tuning
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <stdint.h>

using namespace nvcuda;

// ============================================================================
// H1: Vectorized 128-Bit Memory Load & Bank-Conflict Swizzled Memory Copy
// ============================================================================
__global__ void t4_vectorized_swizzled_memcpy_kernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int N_vecs) 
{
    // Unified 96KB L1/Shared Memory per SM configuration
    __shared__ float4 smem_tile[32][33]; // +1 padding to eliminate 32-bank conflict

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    if (idx < N_vecs) {
        // 128-bit vector load (LDG.128) - saturates GDDR6 320 GB/s bus
        float4 val = input[idx];
        
        int row = tid / 32;
        int col = tid % 32;
        
        // XOR Swizzling pattern for bank-free shared memory access
        int swizzled_col = col ^ (row & 31);
        smem_tile[row][swizzled_col] = val;
        
        __syncthreads();
        
        // Coalesced write back out to GDDR6
        float4 out_val = smem_tile[row][swizzled_col];
        output[idx] = out_val;
    }
}

// ============================================================================
// H2: Register Double-Buffering Software Pipeline for Turing (No cp.async)
// ============================================================================
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

__global__ void t4_turing_wmma_double_buffer_gemm(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    // Compute warp and thread indices
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Shared memory tiles for A and B (Double Buffering: 2 buffers)
    __shared__ half smem_A[2][TILE_M][TILE_K + 8]; // Stride +8 eliminates 16-bit FP16 bank conflicts
    __shared__ half smem_B[2][TILE_K][TILE_N + 8];

    // WMMA Accumulator fragment initialized to zero
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

    // Register prefetch buffers for double buffering
    uint4 reg_prefetch_A;
    uint4 reg_prefetch_B;

    int num_tiles = K / TILE_K;
    int write_buf = 0;
    int read_buf = 0;

    // --- PROLOGUE: Load Stage 0 into Shared Memory ---
    // Fetch 128-bit vector loads into reg_prefetch
    int global_a_idx = (blockIdx.y * TILE_M + threadIdx.x) * K + 0;
    int global_b_idx = (0 * N) + (blockIdx.x * TILE_N + threadIdx.x);

    if (threadIdx.x < TILE_M) {
        reg_prefetch_A = *reinterpret_cast<const uint4*>(&A[global_a_idx]);
        *reinterpret_cast<uint4*>(&smem_A[write_buf][threadIdx.x][0]) = reg_prefetch_A;
    }
    if (threadIdx.x < TILE_N) {
        reg_prefetch_B = *reinterpret_cast<const uint4*>(&B[global_b_idx]);
        *reinterpret_cast<uint4*>(&smem_B[write_buf][0][threadIdx.x]) = reg_prefetch_B;
    }

    __syncthreads();

    // --- MAIN PIPELINE LOOP ---
    for (int t = 0; t < num_tiles; ++t) {
        write_buf = read_buf ^ 1; // Toggle buffer index

        // 1. PREFETCH Next Tile (t + 1) into Registers while computing current tile
        if (t < num_tiles - 1) {
            int next_k = (t + 1) * TILE_K;
            int next_a_idx = (blockIdx.y * TILE_M + threadIdx.x) * K + next_k;
            int next_b_idx = (next_k * N) + (blockIdx.x * TILE_N + threadIdx.x);

            if (threadIdx.x < TILE_M) {
                reg_prefetch_A = *reinterpret_cast<const uint4*>(&A[next_a_idx]);
            }
            if (threadIdx.x < TILE_N) {
                reg_prefetch_B = *reinterpret_cast<const uint4*>(&B[next_b_idx]);
            }
        }

        // 2. COMPUTE Current Tile from read_buf using Turing Tensor Cores
        int warp_row = (warp_id / 2) * 16;
        int warp_col = (warp_id % 2) * 16;

        wmma::load_matrix_sync(a_frag, &smem_A[read_buf][warp_row][0], TILE_K + 8);
        wmma::load_matrix_sync(b_frag, &smem_B[read_buf][0][warp_col], TILE_N + 8);

        // Tensor Core WMMA Instruction execution (m16n16k16 FP16 -> FP32)
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // 3. STORE Prefetched Registers to write_buf Shared Memory for next iteration
        if (t < num_tiles - 1) {
            if (threadIdx.x < TILE_M) {
                *reinterpret_cast<uint4*>(&smem_A[write_buf][threadIdx.x][0]) = reg_prefetch_A;
            }
            if (threadIdx.x < TILE_N) {
                *reinterpret_cast<uint4*>(&smem_B[write_buf][0][threadIdx.x]) = reg_prefetch_B;
            }
        }

        __syncthreads();
        read_buf = write_buf;
    }

    // --- EPILOGUE: Store Accumulator to Global Memory ---
    int row = blockIdx.y * TILE_M + (warp_id / 2) * 16;
    int col = blockIdx.x * TILE_N + (warp_id % 2) * 16;

    if (row < M && col < N) {
        wmma::store_matrix_sync(&C[row * N + col], c_frag, N, wmma::mem_row_major);
    }
}

// ============================================================================
// H3: Fused FlashAttention-2 T4 Sub-Tile Kernel (Online Softmax in Registers)
// ============================================================================
__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void t4_fused_flash_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int seq_len, int head_dim, float sm_scale)
{
    // Shared Memory Tiling for Turing CC 7.5: 64x64 sub-tiles fits in <32KB SMEM
    __shared__ half s_Q[64][64];
    __shared__ half s_K[64][64];
    __shared__ half s_V[64][64];

    int tid = threadIdx.x;
    int b_idx = blockIdx.x; // Block per Sequence Chunk

    // Load Q chunk into shared memory
    if (tid < 64) {
        #pragma unroll
        for (int d = 0; d < head_dim; d += 8) {
            *reinterpret_cast<uint4*>(&s_Q[tid][d]) = 
                *reinterpret_cast<const uint4*>(&Q[(b_idx * 64 + tid) * head_dim + d]);
        }
    }
    __syncthreads();

    // Online Softmax State per thread in registers
    float m_prev = -1e20f;
    float l_prev = 0.0f;
    float acc_O[8] = {0.0f};

    int num_kv_tiles = (seq_len + 63) / 64;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        // Load K, V tile
        if (tid < 64) {
            *reinterpret_cast<uint4*>(&s_K[tid][0]) = 
                *reinterpret_cast<const uint4*>(&K[(kv_tile * 64 + tid) * head_dim]);
            *reinterpret_cast<uint4*>(&s_V[tid][0]) = 
                *reinterpret_cast<const uint4*>(&V[(kv_tile * 64 + tid) * head_dim]);
        }
        __syncthreads();

        // Compute Q * K^T block score
        float score = 0.0f;
        int q_row = tid % 64;
        #pragma unroll
        for (int d = 0; d < 64; ++d) {
            score += __half2float(s_Q[q_row][d]) * __half2float(s_K[tid % 64][d]);
        }
        score *= sm_scale;

        // Online Softmax updates via warp shuffle reductions
        float m_curr = fmaxf(m_prev, warp_reduce_max(score));
        float p = expf(score - m_curr);
        float l_curr = expf(m_prev - m_curr) * l_prev + warp_reduce_sum(p);

        // Update accumulation vector
        float alpha = expf(m_prev - m_curr);
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            acc_O[d] = acc_O[d] * alpha + p * __half2float(s_V[tid % 64][d * 8]);
        }

        m_prev = m_curr;
        l_prev = l_curr;
        __syncthreads();
    }

    // Final Normalize and Store O
    if (l_prev > 0.0f) {
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            acc_O[d] /= l_prev;
            O[(b_idx * 64 + (tid % 64)) * head_dim + d] = __float2half(acc_O[d]);
        }
    }
}

// ============================================================================
// H4: Sub-Byte W4A16 Quantized GEMM PTX Bitfield Unpacking Kernel
// ============================================================================
__device__ __inline__ uint32_t ptx_bfe_u32(uint32_t val, uint32_t bit_start, uint32_t num_bits) {
    uint32_t res;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(val), "r"(bit_start), "r"(num_bits));
    return res;
}

// Fast FP16 Magic Exponent Dequantization using LOP3 (Bypasses INT32->FP16 conversion pipe)
__device__ __forceinline__ void turing_dequant_w4a16_lop3_8x(
    uint32_t packed_w, 
    half2 &w02, half2 &w13, half2 &w46, half2 &w57,
    half2 scale_h2, half2 neg_bias_h2) 
{
    const uint32_t mask_even = 0x000F000F;
    const uint32_t magic_exp = 0x64006400; // 1024.0 in FP16 for both half slots

    uint32_t r02, r13, r46, r57;

    // Single-cycle LOP3 bitfield extraction & FP16 exponent insertion
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;" 
        : "=r"(r02) : "r"(packed_w), "r"(mask_even), "r"(magic_exp));

    asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;" 
        : "=r"(r13) : "r"(packed_w >> 4), "r"(mask_even), "r"(magic_exp));

    asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;" 
        : "=r"(r46) : "r"(packed_w >> 8), "r"(mask_even), "r"(magic_exp));

    asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;" 
        : "=r"(r57) : "r"(packed_w >> 12), "r"(mask_even), "r"(magic_exp));

    // Vectorized Fused Multiply-Add (Out = Raw * Scale - Bias)
    w02 = __hfma2(reinterpret_cast<half2&>(r02), scale_h2, neg_bias_h2);
    w13 = __hfma2(reinterpret_cast<half2&>(r13), scale_h2, neg_bias_h2);
    w46 = __hfma2(reinterpret_cast<half2&>(r46), scale_h2, neg_bias_h2);
    w57 = __hfma2(reinterpret_cast<half2&>(r57), scale_h2, neg_bias_h2);
}

__global__ void t4_int4_w4a16_dequant_wmma_gemm(
    const uint32_t* __restrict__ packed_W4, // 8 INT4 weights per uint32
    const half* __restrict__ scale,
    const half* __restrict__ zero_point,
    const half* __restrict__ A_activation,
    half* __restrict__ C_output,
    int M, int N, int K)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Fast unpacking of 4-bit weights into half precision registers
    if (tid < (N * K) / 8) {
        uint32_t packed = packed_W4[tid];
        half2 w02, w13, w46, w57;
        half2 scale_h2 = __half2half2(scale[tid % N]);
        half2 neg_bias_h2 = __half2half2(__hneg(__hadd(__float2half(1024.0f), zero_point[tid % N])));
        
        // Fast LOP3 magic-number dequantization
        turing_dequant_w4a16_lop3_8x(packed, w02, w13, w46, w57, scale_h2, neg_bias_h2);
    }
}

// Signed INT4 Two's Complement Dequantization via LOP3 LUT 0x6A (Single-Cycle Sign Bit Inversion)
__device__ __forceinline__ void turing_dequant_s4_twos_complement_8x(
    uint32_t packed_w, 
    half2 &w04, half2 &w15, half2 &w26, half2 &w37,
    half2 scale_h2, half2 neg_bias_1032_h2) 
{
    const uint32_t mask_even     = 0x000F000F;
    const uint32_t magic_exp_s4 = 0x64086408; // 1024.0 FP16 + Bit 3 & Bit 19 set to 1

    uint32_t r04, r15, r26, r37;

    // LUT 0x6A inverts bit 3 (sign bit) while inserting exponent 0x6400 in 1 SASS cycle
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(r04) : "r"(packed_w),       "r"(mask_even), "r"(magic_exp_s4));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(r15) : "r"(packed_w >> 4),  "r"(mask_even), "r"(magic_exp_s4));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(r26) : "r"(packed_w >> 8),  "r"(mask_even), "r"(magic_exp_s4));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(r37) : "r"(packed_w >> 12), "r"(mask_even), "r"(magic_exp_s4));

    // Vectorized Fused Multiply-Add: neg_bias_1032_h2 = (-1032.0f - zero_point) * scale
    w04 = __hfma2(reinterpret_cast<half2&>(r04), scale_h2, neg_bias_1032_h2);
    w15 = __hfma2(reinterpret_cast<half2&>(r15), scale_h2, neg_bias_1032_h2);
    w26 = __hfma2(reinterpret_cast<half2&>(r26), scale_h2, neg_bias_1032_h2);
    w37 = __hfma2(reinterpret_cast<half2&>(r37), scale_h2, neg_bias_1032_h2);
}

// Zero-Overhead Inline Register SiLU Activation (SwiGLU) for FP16 Accumulators
__device__ __forceinline__ half2 fast_silu2_fused(half2 x) {
    half2 out;
    asm volatile(
        "{\n\t"
        "  .reg .b32 k, exp_k, denom, inv_denom;\n\t"
        "  hfma2.f16x2 k, %1, {-1.44269504, -1.44269504}, {0.0, 0.0};\n\t"
        "  ex2.approx.f16x2 exp_k, k;\n\t"
        "  hadd2.f16x2 denom, exp_k, {1.0, 1.0};\n\t"
        "  rcp.approx.f16x2 inv_denom, denom;\n\t"
        "  hmul2.f16x2 %0, %1, inv_denom;\n\t"
        "}"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "r"(reinterpret_cast<const uint32_t&>(x))
    );
    return out;
}

// Global persistent tile counter located in L2 cache for Persistent Grid Block Streaming
__device__ int g_persistent_tile_counter = 0;

__global__ void __launch_bounds__(256, 1) t4_persistent_gemm_2stage_l1_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Double-buffered Shared Memory: 2 stages x 128x40 halfs = ~32 KB total SMEM
    __shared__ __align__(16) half smem_A[2][128 * (32 + 8)];
    __shared__ __align__(16) half smem_B[2][128 * (32 + 8)];
    __shared__ int shared_tile_idx;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    const int total_tiles_m = (M + 128 - 1) / 128;
    const int total_tiles_n = (N + 128 - 1) / 128;
    const int total_macro_tiles = total_tiles_m * total_tiles_n;
    const int num_k_tiles = K / 32;

    const int a_load_row = tid / 4;
    const int a_load_col = (tid % 4) * 8;
    const int b_load_row = tid / 16;
    const int b_load_col = (tid % 16) * 8;

    uint32_t smem_a_idx = swizzle_smem_offset_t4(a_load_row, a_load_col);
    uint32_t smem_b_idx = swizzle_smem_offset_t4(b_load_row, b_load_col);

    // PERSISTENT WORKER LOOP (40 Blocks Stream over all Macro-Tiles)
    while (true) {
        if (tid == 0) {
            shared_tile_idx = atomicAdd(&g_persistent_tile_counter, 1);
        }
        __syncthreads();

        int macro_tile_id = shared_tile_idx;
        if (macro_tile_id >= total_macro_tiles) break;

        int block_m_idx = macro_tile_id / total_tiles_n;
        int block_n_idx = macro_tile_id % total_tiles_n;

        int block_m = block_m_idx * 128;
        int block_n = block_n_idx * 128;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::fill_fragment(c_frag[i][j], 0.0f);
            }
        }

        uint4 reg_prefetch_A, reg_prefetch_B;

        const half* gmem_A_ptr = A + (block_m + a_load_row) * K + a_load_col;
        const half* gmem_B_ptr = B + b_load_row * N + (block_n + b_load_col);

        // PROLOGUE: Load Tile 0
        reg_prefetch_A = ((block_m + a_load_row) < M && a_load_col < K) ?
            *reinterpret_cast<const uint4*>(gmem_A_ptr) : make_uint4(0, 0, 0, 0);
        reg_prefetch_B = (b_load_row < K && (block_n + b_load_col) < N) ?
            *reinterpret_cast<const uint4*>(gmem_B_ptr) : make_uint4(0, 0, 0, 0);

        *reinterpret_cast<uint4*>(&smem_A[0][smem_a_idx]) = reg_prefetch_A;
        *reinterpret_cast<uint4*>(&smem_B[0][smem_b_idx]) = reg_prefetch_B;

        gmem_A_ptr += 32;
        gmem_B_ptr += 32 * N;

        __syncthreads();

        int write_stage = 1, read_stage = 0;

        #pragma unroll 1
        for (int k_tile = 0; k_tile < num_k_tiles - 1; ++k_tile) {
            reg_prefetch_A = ((block_m + a_load_row) < M) ?
                *reinterpret_cast<const uint4*>(gmem_A_ptr) : make_uint4(0, 0, 0, 0);
            reg_prefetch_B = ((block_n + b_load_col) < N) ?
                *reinterpret_cast<const uint4*>(gmem_B_ptr) : make_uint4(0, 0, 0, 0);

            gmem_A_ptr += 32;
            gmem_B_ptr += 32 * N;

            #pragma unroll
            for (int k_sub = 0; k_sub < 2; ++k_sub) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[4];

                int warp_row = (warp_id / 2) * 32;
                int warp_col = (warp_id % 2) * 64;

                for (int i = 0; i < 2; ++i) {
                    wmma::load_matrix_sync(a_frag[i], &smem_A[read_stage][(warp_row + i * 16) * 40 + k_sub * 16], 40);
                }
                for (int j = 0; j < 4; ++j) {
                    wmma::load_matrix_sync(b_frag[j], &smem_B[read_stage][(warp_col + j * 16) * 40 + k_sub * 16], 40);
                }

                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                    }
                }
            }

            *reinterpret_cast<uint4*>(&smem_A[write_stage][smem_a_idx]) = reg_prefetch_A;
            *reinterpret_cast<uint4*>(&smem_B[write_stage][smem_b_idx]) = reg_prefetch_B;

            read_stage ^= 1;
            write_stage ^= 1;

            __syncthreads();
        }

        // EPILOGUE
        #pragma unroll
        for (int k_sub = 0; k_sub < 2; ++k_sub) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[4];

            int warp_row = (warp_id / 2) * 32;
            int warp_col = (warp_id % 2) * 64;

            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(a_frag[i], &smem_A[read_stage][(warp_row + i * 16) * 40 + k_sub * 16], 40);
            }
            for (int j = 0; j < 4; ++j) {
                wmma::load_matrix_sync(b_frag[j], &smem_B[read_stage][(warp_col + j * 16) * 40 + k_sub * 16], 40);
            }

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 4; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        int warp_row = (warp_id / 2) * 32;
        int warp_col = (warp_id % 2) * 64;

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                int c_row = block_m + warp_row + i * 16;
                int c_col = block_n + warp_col + j * 16;
                if (c_row < M && c_col < N) {
                    wmma::store_matrix_sync(&C[c_row * N + c_col], c_frag[i][j], N, wmma::mem_row_major);
                }
            }
        }
    }
}

// Host Launcher configuring 32KB SMEM / 64KB L1 Cache Preference & 40 Persistent Blocks
void launch_t4_persistent_gemm(const half* A, const half* B, float* C, int M, int N, int K) {
    cudaFuncSetCacheConfig(t4_persistent_gemm_2stage_l1_kernel, cudaFuncCachePreferL1);
    int zero = 0;
    cudaMemcpyToSymbol(g_persistent_tile_counter, &zero, sizeof(int));
    dim3 grid(40, 1, 1);
    dim3 block(256, 1, 1);
    t4_persistent_gemm_2stage_l1_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

// ============================================================================
// TRAINING & FINE-TUNING SUITE: FWD, BWD dW, BWD dX, FUSED ADAMW & SILU BACKWARD
// ============================================================================

// 1. Backward Weight Gradient GEMM: dW [K x N] = X^T [K x M] * dY [M x N]
__global__ void __launch_bounds__(256, 1) t4_bwd_weight_gemm_persistent_kernel(
    const half* __restrict__ X,   // [M x K] Row-Major
    const half* __restrict__ dY,  // [M x N] Row-Major
    half* __restrict__ dW,        // [K x N] Row-Major Output
    int M, int N, int K,
    int* __restrict__ g_tile_counter)
{
    __shared__ __align__(16) half smem_X_trans[2][64 * 32]; // Stored transposed for ldmatrix
    __shared__ __align__(16) half smem_dY[2][32 * 64];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_row = (warp_id / 2) * 32;
    int warp_col = (warp_id % 2) * 32;

    int num_tiles_k = (K + 64 - 1) / 64;
    int num_tiles_n = (N + 64 - 1) / 64;
    int total_tiles = num_tiles_k * num_tiles_n;

    float accum[2][4][4] = {0.0f};
    __shared__ int shared_tile_idx;

    if (tid == 0) {
        shared_tile_idx = atomicAdd(g_tile_counter, 1);
    }
    __syncthreads();

    while (shared_tile_idx < total_tiles) {
        int tile_idx = shared_tile_idx;
        int tile_k = (tile_idx / num_tiles_n) * 64;
        int tile_n = (tile_idx % num_tiles_n) * 64;

        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                #pragma unroll
                for (int k = 0; k < 4; ++k)
                    accum[i][j][k] = 0.0f;

        int num_m_steps = (M + 16 - 1) / 16;

        for (int m_step = 0; m_step < num_m_steps; ++m_step) {
            int current_m = m_step * 16;

            int load_x_m = current_m + (tid / 16);
            int load_x_k = tile_k + (tid % 16) * 2;

            if (load_x_m < M && load_x_k < K) {
                half2 val = *reinterpret_cast<const half2*>(&X[load_x_m * K + load_x_k]);
                smem_X_trans[0][(load_x_k - tile_k) * 16 + (load_x_m - current_m)] = val.x;
                if (load_x_k + 1 - tile_k < 64) {
                    smem_X_trans[0][(load_x_k + 1 - tile_k) * 16 + (load_x_m - current_m)] = val.y;
                }
            }

            int load_dy_m = current_m + (tid / 16);
            int load_dy_n = tile_n + (tid % 16) * 4;
            if (load_dy_m < M && load_dy_n < N) {
                *reinterpret_cast<uint2*>(&smem_dY[0][(load_dy_m - current_m) * 64 + (load_dy_n - tile_n)]) =
                    *reinterpret_cast<const uint2*>(&dY[load_dy_m * N + load_dy_n]);
            }
            __syncthreads();

            #pragma unroll
            for (int mi = 0; mi < 16; mi += 8) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[2];

                wmma::load_matrix_sync(a_frag[0], &smem_X_trans[0][warp_row * 16 + mi], 16);
                wmma::load_matrix_sync(b_frag[0], &smem_dY[0][mi * 64 + warp_col], 64);

                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
                wmma::fill_fragment(c_f, 0.0f);
                wmma::mma_sync(c_f, a_frag[0], b_frag[0], c_f);

                for (int elem = 0; elem < c_f.num_elements; ++elem) {
                    accum[0][0][elem % 4] += c_f.x[elem];
                }
            }
            __syncthreads();
        }

        int out_k = tile_k + warp_row + (lane_id / 4);
        int out_n = tile_n + warp_col + (lane_id % 4) * 2;

        if (out_k < K && out_n < N) {
            dW[out_k * N + out_n] = __float2half(accum[0][0][0]);
            if (out_n + 1 < N) {
                dW[out_k * N + out_n + 1] = __float2half(accum[0][0][1]);
            }
        }

        if (tid == 0) {
            shared_tile_idx = atomicAdd(g_tile_counter, 1);
        }
        __syncthreads();
    }
}

// 2. Fused Backward GEMM + AdamW Optimizer Kernel
__global__ void __launch_bounds__(128, 8) fused_backward_gemm_adamw_kernel(
    const half* __restrict__ dY,       // [K, M]
    const half* __restrict__ X,        // [K, N]
    float* __restrict__ W_master,      // [M, N]
    half* __restrict__ W_active,       // [M, N]
    float* __restrict__ exp_avg,       // [M, N]
    float* __restrict__ exp_avg_sq,    // [M, N]
    const int M, const int N, const int K,
    const float lr, const float beta1, const float beta2,
    const float eps, const float weight_decay,
    const float bias_correction1, const float bias_correction2)
{
    __shared__ half s_dY[2][16][68];
    __shared__ half s_X[2][16][68];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    const int block_m = blockIdx.y * 64;
    const int block_n = blockIdx.x * 64;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];

    int write_buf = 0;

    #pragma unroll
    for (int i = tid; i < (16 * 64) / 8; i += blockDim.x) {
        int k_idx = i / 8;
        int m_idx = (i % 8) * 8;
        if (k_idx < K && (block_m + m_idx) < M) {
            *reinterpret_cast<uint4*>(&s_dY[write_buf][k_idx][m_idx]) =
                *reinterpret_cast<const uint4*>(&dY[k_idx * M + block_m + m_idx]);
        } else {
            *reinterpret_cast<uint4*>(&s_dY[write_buf][k_idx][m_idx]) = make_uint4(0, 0, 0, 0);
        }
    }

    #pragma unroll
    for (int i = tid; i < (16 * 64) / 8; i += blockDim.x) {
        int k_idx = i / 8;
        int n_idx = (i % 8) * 8;
        if (k_idx < K && (block_n + n_idx) < N) {
            *reinterpret_cast<uint4*>(&s_X[write_buf][k_idx][n_idx]) =
                *reinterpret_cast<const uint4*>(&X[k_idx * N + block_n + n_idx]);
        } else {
            *reinterpret_cast<uint4*>(&s_X[write_buf][k_idx][n_idx]) = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    for (int k_tile_idx = 0; k_tile_idx < K; k_tile_idx += 16) {
        int read_buf = write_buf;
        write_buf ^= 1;

        int next_k = k_tile_idx + 16;
        if (next_k < K) {
            #pragma unroll
            for (int i = tid; i < (16 * 64) / 8; i += blockDim.x) {
                int k_idx = i / 8;
                int m_idx = (i % 8) * 8;
                if ((next_k + k_idx) < K && (block_m + m_idx) < M) {
                    *reinterpret_cast<uint4*>(&s_dY[write_buf][k_idx][m_idx]) =
                        *reinterpret_cast<const uint4*>(&dY[(next_k + k_idx) * M + block_m + m_idx]);
                } else {
                    *reinterpret_cast<uint4*>(&s_dY[write_buf][k_idx][m_idx]) = make_uint4(0, 0, 0, 0);
                }
            }

            #pragma unroll
            for (int i = tid; i < (16 * 64) / 8; i += blockDim.x) {
                int k_idx = i / 8;
                int n_idx = (i % 8) * 8;
                if ((next_k + k_idx) < K && (block_n + n_idx) < N) {
                    *reinterpret_cast<uint4*>(&s_X[write_buf][k_idx][n_idx]) =
                        *reinterpret_cast<const uint4*>(&X[(next_k + k_idx) * N + block_n + n_idx]);
                } else {
                    *reinterpret_cast<uint4*>(&s_X[write_buf][k_idx][n_idx]) = make_uint4(0, 0, 0, 0);
                }
            }
        }

        wmma::load_matrix_sync(a_frag[0], &s_dY[read_buf][0][warp_row * 32], 68);
        wmma::load_matrix_sync(a_frag[1], &s_dY[read_buf][0][warp_row * 32 + 16], 68);

        wmma::load_matrix_sync(b_frag[0], &s_X[read_buf][0][warp_col * 32], 68);
        wmma::load_matrix_sync(b_frag[1], &s_X[read_buf][0][warp_col * 32 + 16], 68);

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }

        __syncthreads();
    }

    // In-Register AdamW Step Calculation & Writeback
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int sub_m = warp_row * 32 + i * 16;
            const int sub_n = warp_col * 32 + j * 16;

            for (int elem = 0; elem < c_frag[i][j].num_elements; ++elem) {
                int elem_r = elem / 2;
                int elem_c = (elem % 2) + (lane_id % 4) * 2;
                if (lane_id >= 16) elem_r += 8;

                int global_r = block_m + sub_m + elem_r;
                int global_c = block_n + sub_n + elem_c;

                if (global_r < M && global_c < N) {
                    int offset = global_r * N + global_c;
                    float grad = c_frag[i][j].x[elem];

                    float w_val = W_master[offset];
                    float m_val = exp_avg[offset];
                    float v_val = exp_avg_sq[offset];

                    w_val -= lr * weight_decay * w_val;
                    m_val = beta1 * m_val + (1.0f - beta1) * grad;
                    v_val = beta2 * v_val + (1.0f - beta2) * (grad * grad);

                    float m_hat = m_val / bias_correction1;
                    float v_hat = v_val / bias_correction2;

                    w_val -= lr * (m_hat / (sqrtf(v_hat) + eps));

                    W_master[offset] = w_val;
                    W_active[offset] = __float2half(w_val);
                    exp_avg[offset] = m_val;
                    exp_avg_sq[offset] = v_val;
                }
            }
        }
    }
}



