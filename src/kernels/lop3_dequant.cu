// Tesla T4 LOP3 Fast Dequantization (0xEA Unsigned / 0x6A Signed)
#include "lop3_dequant.h"

__device__ __forceinline__ uint32_t pack_half_dup_u32(half val) {
    uint16_t bits = __half_as_ushort(val);
    return ((uint32_t)bits << 16) | (uint32_t)bits;
}

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

    float s_f = __half2float(s);
    float z_f = __half2float(z);
    float bias_f = (-1024.0f - z_f) * s_f;
    half bias_h = __float2half(bias_f);

    uint32_t scale_32 = pack_half_dup_u32(s);
    uint32_t neg_bias_32 = pack_half_dup_u32(bias_h);

    uint32_t raw_04, raw_15, raw_26, raw_37;
    lop3_unpack_u4_ptx(W, raw_04, raw_15, raw_26, raw_37, scale_32, neg_bias_32);

    int out_offset = idx * 8;
    uint16_t* out_ptr = reinterpret_cast<uint16_t*>(output_fp16 + out_offset);

    out_ptr[0] = (uint16_t)(raw_04 & 0xFFFF);
    out_ptr[1] = (uint16_t)(raw_15 & 0xFFFF);
    out_ptr[2] = (uint16_t)(raw_26 & 0xFFFF);
    out_ptr[3] = (uint16_t)(raw_37 & 0xFFFF);
    out_ptr[4] = (uint16_t)(raw_04 >> 16);
    out_ptr[5] = (uint16_t)(raw_15 >> 16);
    out_ptr[6] = (uint16_t)(raw_26 >> 16);
    out_ptr[7] = (uint16_t)(raw_37 >> 16);
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

    float s_f = __half2float(s);
    float z_f = __half2float(z);
    float bias_f = (-1032.0f - z_f) * s_f;
    half bias_h = __float2half(bias_f);

    uint32_t scale_32 = pack_half_dup_u32(s);
    uint32_t neg_bias_1032_32 = pack_half_dup_u32(bias_h);

    uint32_t raw_04, raw_15, raw_26, raw_37;
    lop3_unpack_s4_ptx(W, raw_04, raw_15, raw_26, raw_37, scale_32, neg_bias_1032_32);

    int out_offset = idx * 8;
    uint16_t* out_ptr = reinterpret_cast<uint16_t*>(output_fp16 + out_offset);

    out_ptr[0] = (uint16_t)(raw_04 & 0xFFFF);
    out_ptr[1] = (uint16_t)(raw_15 & 0xFFFF);
    out_ptr[2] = (uint16_t)(raw_26 & 0xFFFF);
    out_ptr[3] = (uint16_t)(raw_37 & 0xFFFF);
    out_ptr[4] = (uint16_t)(raw_04 >> 16);
    out_ptr[5] = (uint16_t)(raw_15 >> 16);
    out_ptr[6] = (uint16_t)(raw_26 >> 16);
    out_ptr[7] = (uint16_t)(raw_37 >> 16);
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
