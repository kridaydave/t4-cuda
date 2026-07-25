#include "fused_w4a16_gemm.h"
#include "lop3_dequant.h"
#include <cuda_fp16.h>

__device__ __forceinline__ uint32_t pack_half_dup_u32_fused(half val) {
    uint16_t bits = __half_as_ushort(val);
    return ((uint32_t)bits << 16) | (uint32_t)bits;
}

// -------------------------------------------------------------------------
// Fused Unsigned INT4 GEMV Kernel for Small M (M = 1 .. 8)
// -------------------------------------------------------------------------
// Grid: (N / 4, M)
// Block: (256 threads) -> 8 warps work on 4 columns of output
__global__ void fused_w4a16_gemv_u4_kernel(
    const half* __restrict__ A,          // (M x K)
    const uint32_t* __restrict__ W_packed,// (K/8 x N)
    const half* __restrict__ scale,      // (1 x N)
    const half* __restrict__ zero_point, // (1 x N)
    half* __restrict__ C,                // (M x N)
    int M, int N, int K)
{
    int col = blockIdx.x * 4 + (threadIdx.x % 4);
    int m = blockIdx.y;

    if (col >= N || m >= M) return;

    int tid_k = threadIdx.x / 4; // 0 .. 63
    int stride_k = blockDim.x / 4; // 64 threads working along K per column

    half s = scale[col];
    half z = zero_point[col];

    float s_f = __half2float(s);
    float z_f = __half2float(z);
    float bias_f = (-1024.0f - z_f) * s_f;
    half bias_h = __float2half(bias_f);

    uint32_t scale_32 = pack_half_dup_u32_fused(s);
    uint32_t neg_bias_32 = pack_half_dup_u32_fused(bias_h);

    const half* A_row = A + m * K;
    const uint32_t* W_col = W_packed + col; // Column stride N uint32s

    float accum_f = 0.0f;
    int k_uint32_total = K / 8;

    for (int k_idx = tid_k; k_idx < k_uint32_total; k_idx += stride_k) {
        // 1. Load packed uint32 weight (8 INT4 weights)
        uint32_t packed_w = W_col[k_idx * N];

        // 2. LOP3 0xEA single-cycle register dequantization
        uint32_t raw_04, raw_15, raw_26, raw_37;
        lop3_unpack_u4_ptx(packed_w, raw_04, raw_15, raw_26, raw_37, scale_32, neg_bias_32);

        // 3. Load 8 FP16 activations corresponding to these K positions
        const half2* A_h2 = reinterpret_cast<const half2*>(A_row + k_idx * 8);
        half2 a_04 = A_h2[0]; // (a0, a4)
        half2 a_15 = A_h2[1]; // (a1, a5)
        half2 a_26 = A_h2[2]; // (a2, a6)
        half2 a_37 = A_h2[3]; // (a3, a7)

        // 4. Multiply-accumulate in FP16/FP32
        half2 prod04 = __hmul2(reinterpret_cast<const half2&>(raw_04), a_04);
        half2 prod15 = __hmul2(reinterpret_cast<const half2&>(raw_15), a_15);
        half2 prod26 = __hmul2(reinterpret_cast<const half2&>(raw_26), a_26);
        half2 prod37 = __hmul2(reinterpret_cast<const half2&>(raw_37), a_37);

        accum_f += __half2float(prod04.x) + __half2float(prod04.y);
        accum_f += __half2float(prod15.x) + __half2float(prod15.y);
        accum_f += __half2float(prod26.x) + __half2float(prod26.y);
        accum_f += __half2float(prod37.x) + __half2float(prod37.y);
    }

    // Parallel reduction across threads assigned to the same column
    #pragma unroll
    for (int offset = 16; offset >= 4; offset /= 2) {
        accum_f += __shfl_xor_sync(0xFFFFFFFF, accum_f, offset);
    }

    // Shared memory reduction across warps
    __shared__ float smem_red[64][4]; // 16 warps x 4 cols
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int col_in_warp = lane_id % 4;

    if (lane_id < 4) {
        smem_red[warp_id][col_in_warp] = accum_f;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < 4) {
        float final_sum = 0.0f;
        int num_warps = blockDim.x / 32;
        for (int w = 0; w < num_warps; w++) {
            final_sum += smem_red[w][lane_id];
        }
        int final_col = blockIdx.x * 4 + lane_id;
        if (final_col < N) {
            C[m * N + final_col] = __float2half(final_sum);
        }
    }
}

// -------------------------------------------------------------------------
// Fused Signed INT4 (S4A16) GEMV Kernel for Small M (M = 1 .. 8)
// -------------------------------------------------------------------------
__global__ void fused_w4a16_gemv_s4_kernel(
    const half* __restrict__ A,          // (M x K)
    const uint32_t* __restrict__ W_packed,// (K/8 x N)
    const half* __restrict__ scale,      // (1 x N)
    const half* __restrict__ zero_point, // (1 x N)
    half* __restrict__ C,                // (M x N)
    int M, int N, int K)
{
    int col = blockIdx.x * 4 + (threadIdx.x % 4);
    int m = blockIdx.y;

    if (col >= N || m >= M) return;

    int tid_k = threadIdx.x / 4;
    int stride_k = blockDim.x / 4;

    half s = scale[col];
    half z = zero_point[col];

    float s_f = __half2float(s);
    float z_f = __half2float(z);
    float bias_f = (-1032.0f - z_f) * s_f;
    half bias_h = __float2half(bias_f);

    uint32_t scale_32 = pack_half_dup_u32_fused(s);
    uint32_t neg_bias_1032_32 = pack_half_dup_u32_fused(bias_h);

    const half* A_row = A + m * K;
    const uint32_t* W_col = W_packed + col;

    float accum_f = 0.0f;
    int k_uint32_total = K / 8;

    for (int k_idx = tid_k; k_idx < k_uint32_total; k_idx += stride_k) {
        uint32_t packed_w = W_col[k_idx * N];

        // LOP3 0x6A single-cycle sign-flip + exponent injection
        uint32_t raw_04, raw_15, raw_26, raw_37;
        lop3_unpack_s4_ptx(packed_w, raw_04, raw_15, raw_26, raw_37, scale_32, neg_bias_1032_32);

        const half2* A_h2 = reinterpret_cast<const half2*>(A_row + k_idx * 8);
        half2 a_04 = A_h2[0];
        half2 a_15 = A_h2[1];
        half2 a_26 = A_h2[2];
        half2 a_37 = A_h2[3];

        half2 prod04 = __hmul2(reinterpret_cast<const half2&>(raw_04), a_04);
        half2 prod15 = __hmul2(reinterpret_cast<const half2&>(raw_15), a_15);
        half2 prod26 = __hmul2(reinterpret_cast<const half2&>(raw_26), a_26);
        half2 prod37 = __hmul2(reinterpret_cast<const half2&>(raw_37), a_37);

        accum_f += __half2float(prod04.x) + __half2float(prod04.y);
        accum_f += __half2float(prod15.x) + __half2float(prod15.y);
        accum_f += __half2float(prod26.x) + __half2float(prod26.y);
        accum_f += __half2float(prod37.x) + __half2float(prod37.y);
    }

    #pragma unroll
    for (int offset = 16; offset >= 4; offset /= 2) {
        accum_f += __shfl_xor_sync(0xFFFFFFFF, accum_f, offset);
    }

    __shared__ float smem_red[64][4];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int col_in_warp = lane_id % 4;

    if (lane_id < 4) {
        smem_red[warp_id][col_in_warp] = accum_f;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < 4) {
        float final_sum = 0.0f;
        int num_warps = blockDim.x / 32;
        for (int w = 0; w < num_warps; w++) {
            final_sum += smem_red[w][lane_id];
        }
        int final_col = blockIdx.x * 4 + lane_id;
        if (final_col < N) {
            C[m * N + final_col] = __float2half(final_sum);
        }
    }
}

// Host Launcher Wrappers
void launch_fused_w4a16_gemm_u4(
    const half* d_A,
    const uint32_t* d_W_packed,
    const half* d_scale,
    const half* d_zero,
    half* d_C,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 grid((N + 3) / 4, M);
    dim3 block(256);
    fused_w4a16_gemv_u4_kernel<<<grid, block, 0, stream>>>(
        d_A, d_W_packed, d_scale, d_zero, d_C, M, N, K);
}

void launch_fused_w4a16_gemm_s4(
    const half* d_A,
    const uint32_t* d_W_packed,
    const half* d_scale,
    const half* d_zero,
    half* d_C,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 grid((N + 3) / 4, M);
    dim3 block(256);
    fused_w4a16_gemv_s4_kernel<<<grid, block, 0, stream>>>(
        d_A, d_W_packed, d_scale, d_zero, d_C, M, N, K);
}
