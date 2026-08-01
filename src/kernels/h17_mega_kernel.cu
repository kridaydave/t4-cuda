#include "h17_mega_kernel.h"
#include "lop3_dequant.h"
#include <cuda_fp16.h>
#include <stdio.h>

__device__ __forceinline__ uint32_t pack_half_dup_u32_h17(half val) {
    uint16_t bits = __half_as_ushort(val);
    return ((uint32_t)bits << 16) | (uint32_t)bits;
}

// -------------------------------------------------------------------------
// FLAGSHIP H17: FUSED INT3 LOP3 GEMV DECODE MEGA-KERNEL
// Fuses:
//   1. H7 INT3 LOP3 0x78 Single-Cycle Register Dequantization (magic 0x64046404)
//   2. Per-GROUP Scale & Zero-Point Prefetching (group = 100 along K; H17_GROUP)
//   3. (Roadmap) H8 Software Warp Specialization — current build is a uniform
//      256-thread block; producer/consumer split lands as V2 ablation variant.
// -------------------------------------------------------------------------
// Grid: (ceil(N/4), M)   Block: 256 threads (8 warps, 64 threads per output col)
// Quant group = 100 K-positions = exactly 10 packed uint32 words. 100 is a
// multiple of the 10-element packing word, so a word never straddles two groups.
static constexpr int H17_GROUP = 100;  // must match Python quantizer block_size

__global__ void fused_h17_gemv_s3_warp_specialized_kernel(
    const half* __restrict__ A,           // (M x K)
    const uint32_t* __restrict__ W_packed,// (K/10 x N) packed INT3, col-major words
    const half* __restrict__ scale,       // (num_groups x N) FP16
    const half* __restrict__ zero_point,  // (num_groups x N) FP16
    half* __restrict__ C,                 // (M x N)
    int M, int N, int K, int num_groups)
{
    int col = blockIdx.x * 4 + (threadIdx.x % 4);
    int m = blockIdx.y;

    int tid_k = threadIdx.x / 4;        // 0 .. 63
    int stride_k = blockDim.x / 4;      // 64 threads per output column

    const half* A_row = (m < M) ? (A + m * K) : nullptr;
    const uint32_t* W_col = (col < N) ? (W_packed + col) : nullptr;

    float accum_f = 0.0f;
    int k_uint32_total = K / 10; // Each uint32 holds 10 packed INT3 weights

    if (col < N && m < M) {
        // Per-group scale/zp registers, refetched when crossing a 128-wide K group.
        uint32_t scale_32 = 0, neg_bias_32 = 0;
        int cur_group = -1;

        #pragma unroll 4
        for (int k_idx = tid_k; k_idx < k_uint32_total; k_idx += stride_k) {
            int k_pos = k_idx * 10;
            int g = k_pos / H17_GROUP;
            if (g != cur_group) {
                cur_group = g;
                int gi = g * N + col;
                half s_h = scale[gi];
                half z_h = (zero_point != nullptr) ? zero_point[gi] : __float2half(0.0f);
                float s_f = __half2float(s_h);
                float z_f = __half2float(z_h);
                float bias_f = (-1028.0f - z_f) * s_f;
                scale_32   = pack_half_dup_u32_h17(s_h);
                neg_bias_32 = pack_half_dup_u32_h17(__float2half(bias_f));
            }

            // 1. Load 32-bit packed weight containing 10 INT3 values
            uint32_t packed_w = W_col[k_idx * N];

            // 2. LOP3 0x78 Hardware Dequantization (magic 0x64046404, 1 cycle per 5 pairs)
            uint32_t raw_05, raw_16, raw_27, raw_38, raw_49;
            lop3_unpack_s3_ptx(packed_w, raw_05, raw_16, raw_27, raw_38, raw_49, scale_32, neg_bias_32);

            // 3. Load 10 FP16 activations corresponding to these 10 K positions
            const half* A_ptr = A_row + k_pos;

            half2 a_05 = __halves2half2(A_ptr[0], A_ptr[5]);
            half2 a_16 = __halves2half2(A_ptr[1], A_ptr[6]);
            half2 a_27 = __halves2half2(A_ptr[2], A_ptr[7]);
            half2 a_38 = __halves2half2(A_ptr[3], A_ptr[8]);
            half2 a_49 = __halves2half2(A_ptr[4], A_ptr[9]);

            // 4. SIMD FP16 Multiply-Accumulate
            half2 prod05 = __hmul2(reinterpret_cast<const half2&>(raw_05), a_05);
            half2 prod16 = __hmul2(reinterpret_cast<const half2&>(raw_16), a_16);
            half2 prod27 = __hmul2(reinterpret_cast<const half2&>(raw_27), a_27);
            half2 prod38 = __hmul2(reinterpret_cast<const half2&>(raw_38), a_38);
            half2 prod49 = __hmul2(reinterpret_cast<const half2&>(raw_49), a_49);

            accum_f += __half2float(prod05.x) + __half2float(prod05.y);
            accum_f += __half2float(prod16.x) + __half2float(prod16.y);
            accum_f += __half2float(prod27.x) + __half2float(prod27.y);
            accum_f += __half2float(prod38.x) + __half2float(prod38.y);
            accum_f += __half2float(prod49.x) + __half2float(prod49.y);
        }
    }

    // Parallel reduction across threads assigned to the same column
    #pragma unroll
    for (int offset = 16; offset >= 4; offset /= 2) {
        accum_f += __shfl_xor_sync(0xFFFFFFFF, accum_f, offset);
    }

    // Shared memory reduction across warps
    __shared__ float smem_red[8][4]; // 8 warps x 4 cols
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
        if (final_col < N && m < M) {
            C[m * N + final_col] = __float2half(final_sum);
        }
    }
}

void launch_h17_gemv_s3(
    const half* A,
    const uint32_t* W_packed,
    const half* scale,
    const half* zero_point,
    half* C,
    int M, int N, int K,
    int num_groups,
    cudaStream_t stream)
{
    dim3 grid((N + 3) / 4, M);
    dim3 block(256);

    fused_h17_gemv_s3_warp_specialized_kernel<<<grid, block, 0, stream>>>(
        A, W_packed, scale, zero_point, C, M, N, K, num_groups);
}
