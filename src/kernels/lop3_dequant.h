#ifndef LOP3_DEQUANT_H
#define LOP3_DEQUANT_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __CUDACC__
// Pure PTX Unsigned INT4 Dequantization (0xEA + fma.rn.f16x2)
__device__ __forceinline__ void lop3_unpack_u4_ptx(
    uint32_t W, 
    uint32_t &raw_04, uint32_t &raw_15, uint32_t &raw_26, uint32_t &raw_37,
    uint32_t scale_32, uint32_t neg_bias_32) 
{
    const uint32_t mask_even = 0x000F000F;
    const uint32_t magic_exp = 0x64006400; // 1024.0 FP16

    // 1. Bitwise LOP3 LUT 0xEA: (A & B) | C  [A=W, B=mask, C=magic_exp]
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(raw_04) : "r"(W),       "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(raw_15) : "r"(W >> 4),  "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(raw_26) : "r"(W >> 8),  "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(raw_37) : "r"(W >> 12), "r"(mask_even), "r"(magic_exp));

    // 2. Hardware SIMD FP16 FMA: (raw * scale) + neg_bias (in-out constraint '+r')
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_04) : "r"(scale_32), "r"(neg_bias_32));
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_15) : "r"(scale_32), "r"(neg_bias_32));
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_26) : "r"(scale_32), "r"(neg_bias_32));
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_37) : "r"(scale_32), "r"(neg_bias_32));
}

// Pure PTX Signed INT4 Dequantization (0x78 + fma.rn.f16x2)
__device__ __forceinline__ void lop3_unpack_s4_ptx(
    uint32_t W, 
    uint32_t &raw_04, uint32_t &raw_15, uint32_t &raw_26, uint32_t &raw_37,
    uint32_t scale_32, uint32_t neg_bias_1032_32) 
{
    const uint32_t mask_even = 0x000F000F;
    const uint32_t magic_exp = 0x64086408; // 1032.0 FP16 x2: exponent + bit3 set for XOR-flip of sign

    // 1. Bitwise LOP3 LUT 0x6A: (A^C)&B | C&~B  [A=W, B=mask, C=magic_s4=0x64086408]
    //    When b=1 (nibble bit): take (a XOR c), flipping bit3 since c's bit3=1
    //    When b=0 (non-nibble): take c (exponent bits pass through)
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(raw_04) : "r"(W),       "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(raw_15) : "r"(W >> 4),  "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(raw_26) : "r"(W >> 8),  "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(raw_37) : "r"(W >> 12), "r"(mask_even), "r"(magic_exp));

    // 2. Hardware SIMD FP16 FMA: (raw * scale) + neg_bias_1032 (in-out constraint '+r')
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_04) : "r"(scale_32), "r"(neg_bias_1032_32));
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_15) : "r"(scale_32), "r"(neg_bias_1032_32));
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_26) : "r"(scale_32), "r"(neg_bias_1032_32));
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_37) : "r"(scale_32), "r"(neg_bias_1032_32));
}
#endif // __CUDACC__

void launch_lop3_dequant_u4(
    const uint32_t* d_packed, half* d_output, const half* d_scale, const half* d_zero, int num_uint32s, cudaStream_t stream = 0);

void launch_lop3_dequant_s4(
    const uint32_t* d_packed, half* d_output, const half* d_scale, const half* d_zero, int num_uint32s, cudaStream_t stream = 0);

#endif // LOP3_DEQUANT_H
