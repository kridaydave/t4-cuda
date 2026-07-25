#include "lop3_dequant.h"

__global__ void lop3_dequant_u4_kernel(
    const uint32_t* __restrict__ packed_weights,
    half* __restrict__ output_fp16,
    const half* __restrict__ scale,
    const half* __restrict__ zero_point,
    int num_uint32s)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_uint32s) return;

    uint32_t W = packed_weights[idx];
    half s = scale[idx];
    half z = zero_point[idx];

    half2 scale_h2 = __half2half2(s);
    half half1024 = __float2half(-1024.0f);
    half bias_val = __hmul(__hsub(half1024, z), s);
    half2 neg_bias_h2 = __half2half2(bias_val);

    half2 h2_04, h2_15, h2_26, h2_37;
    lop3_unpack_u4_to_4_half2(W, h2_04, h2_15, h2_26, h2_37, scale_h2, neg_bias_h2);

    int out_offset = idx * 8;
    half* out_ptr = output_fp16 + out_offset;

    out_ptr[0] = __low2half(h2_04);
    out_ptr[1] = __low2half(h2_15);
    out_ptr[2] = __low2half(h2_26);
    out_ptr[3] = __low2half(h2_37);
    out_ptr[4] = __high2half(h2_04);
    out_ptr[5] = __high2half(h2_15);
    out_ptr[6] = __high2half(h2_26);
    out_ptr[7] = __high2half(h2_37);
}

__global__ void lop3_dequant_s4_kernel(
    const uint32_t* __restrict__ packed_weights,
    half* __restrict__ output_fp16,
    const half* __restrict__ scale,
    const half* __restrict__ zero_point,
    int num_uint32s)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_uint32s) return;

    uint32_t W = packed_weights[idx];
    half s = scale[idx];
    half z = zero_point[idx];

    half2 scale_h2 = __half2half2(s);
    half half1032 = __float2half(-1032.0f);
    half bias_val = __hmul(__hsub(half1032, z), s);
    half2 neg_bias_1032_h2 = __half2half2(bias_val);

    half2 h2_04, h2_15, h2_26, h2_37;
    lop3_unpack_s4_to_4_half2(W, h2_04, h2_15, h2_26, h2_37, scale_h2, neg_bias_1032_h2);

    int out_offset = idx * 8;
    half* out_ptr = output_fp16 + out_offset;

    out_ptr[0] = __low2half(h2_04);
    out_ptr[1] = __low2half(h2_15);
    out_ptr[2] = __low2half(h2_26);
    out_ptr[3] = __low2half(h2_37);
    out_ptr[4] = __high2half(h2_04);
    out_ptr[5] = __high2half(h2_15);
    out_ptr[6] = __high2half(h2_26);
    out_ptr[7] = __high2half(h2_37);
}

void launch_lop3_dequant_u4(
    const uint32_t* d_packed, half* d_output, const half* d_scale, const half* d_zero, int num_uint32s, cudaStream_t stream)
{
    int threads_per_block = 256;
    int blocks = (num_uint32s + threads_per_block - 1) / threads_per_block;
    lop3_dequant_u4_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_packed, d_output, d_scale, d_zero, num_uint32s);
}

void launch_lop3_dequant_s4(
    const uint32_t* d_packed, half* d_output, const half* d_scale, const half* d_zero, int num_uint32s, cudaStream_t stream)
{
    int threads_per_block = 256;
    int blocks = (num_uint32s + threads_per_block - 1) / threads_per_block;
    lop3_dequant_s4_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_packed, d_output, d_scale, d_zero, num_uint32s);
}
