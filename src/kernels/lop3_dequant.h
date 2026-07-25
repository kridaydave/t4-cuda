#ifndef LOP3_DEQUANT_H
#define LOP3_DEQUANT_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __CUDACC__
// Pure PTX Unsigned INT4 Dequantization (0xF2 + fma.rn.f16x2)
__device__ __forceinline__ void lop3_unpack_u4_ptx(
    uint32_t W, 
    uint32_t &raw_04, uint32_t &raw_15, uint32_t &raw_26, uint32_t &raw_37,
    uint32_t scale_32, uint32_t neg_bias_32) 
{
    const uint32_t mask_even = 0x000F000F;
    const uint32_t magic_exp = 0x64006400; // 1024.0 FP16

    // 1. Bitwise LOP3 LUT 0xF2: (A & B) | C
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xF2;" : "=r"(raw_04) : "r"(W),       "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xF2;" : "=r"(raw_15) : "r"(W >> 4),  "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xF2;" : "=r"(raw_26) : "r"(W >> 8),  "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xF2;" : "=r"(raw_37) : "r"(W >> 12), "r"(mask_even), "r"(magic_exp));

    // 2. Hardware SIMD FP16 FMA: (raw * scale) + neg_bias
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(raw_04) : "r"(raw_04), "r"(scale_32), "r"(neg_bias_32));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(raw_15) : "r"(raw_15), "r"(scale_32), "r"(neg_bias_32));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(raw_26) : "r"(raw_26), "r"(scale_32), "r"(neg_bias_32));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(raw_37) : "r"(raw_37), "r"(scale_32), "r"(neg_bias_32));
}

// Pure PTX Signed INT4 Dequantization (0x78 + fma.rn.f16x2)
__device__ __forceinline__ void lop3_unpack_s4_ptx(
    uint32_t W, 
    uint32_t &raw_04, uint32_t &raw_15, uint32_t &raw_26, uint32_t &raw_37,
    uint32_t scale_32, uint32_t neg_bias_1032_32) 
{
    const uint32_t mask_even = 0x000F000F;
    const uint32_t magic_exp = 0x64006400; // Exponent 1024.0 FP16

    // 1. Bitwise LOP3 LUT 0x78: Invert sign bit 3 & inject exponent
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x78;" : "=r"(raw_04) : "r"(W),       "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x78;" : "=r"(raw_15) : "r"(W >> 4),  "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x78;" : "=r"(raw_26) : "r"(W >> 8),  "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x78;" : "=r"(raw_37) : "r"(W >> 12), "r"(mask_even), "r"(magic_exp));

    // 2. Hardware SIMD FP16 FMA: (raw * scale) + neg_bias_1032
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(raw_04) : "r"(raw_04), "r"(scale_32), "r"(neg_bias_1032_32));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(raw_15) : "r"(raw_15), "r"(scale_32), "r"(neg_bias_1032_32));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(raw_26) : "r"(raw_26), "r"(scale_32), "r"(neg_bias_1032_32));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(raw_37) : "r"(raw_37), "r"(scale_32), "r"(neg_bias_1032_32));
}
#endif // __CUDACC__

void launch_lop3_dequant_u4(
    const uint32_t* d_packed, half* d_output, const half* d_scale, const half* d_zero, int num_uint32s, cudaStream_t stream = 0);

void launch_lop3_dequant_s4(
    const uint32_t* d_packed, half* d_output, const half* d_scale, const half* d_zero, int num_uint32s, cudaStream_t stream = 0);

#endif // LOP3_DEQUANT_H
