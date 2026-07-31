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

// Pure PTX Signed INT4 Dequantization (0x6A + fma.rn.f16x2)
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
// Pure PTX Signed INT3 Dequantization (0xCA + fma.rn.f16x2)
__device__ __forceinline__ void lop3_unpack_s3_ptx(
    uint32_t W, 
    uint32_t &raw_05, uint32_t &raw_16, uint32_t &raw_27, uint32_t &raw_38, uint32_t &raw_49,
    uint32_t scale_32, uint32_t neg_bias_1028_32) 
{
    const uint32_t mask_even_s3 = 0x00070007;
    const uint32_t magic_exp_s3 = 0x64046404; // 1024.0 FP16 with bit2 set (0x6404) for sign-inversion

    // Extract 5 pairs of 3-bit indices (lower half: elem k, upper half: elem k+5)
    uint32_t W05 = (W & 0x0007) | (((W >> 15) & 0x0007) << 16);
    uint32_t W16 = ((W >> 3) & 0x0007) | (((W >> 18) & 0x0007) << 16);
    uint32_t W27 = ((W >> 6) & 0x0007) | (((W >> 21) & 0x0007) << 16);
    uint32_t W38 = ((W >> 9) & 0x0007) | (((W >> 24) & 0x0007) << 16);
    uint32_t W49 = ((W >> 12) & 0x0007) | (((W >> 27) & 0x0007) << 16);

    // 1. Bitwise LOP3 LUT 0x6A: (B & (A ^ C)) | (~B & C)
    //    When B=1 (3-bit mask): output (A ^ C), inverting bit 2 since C's bit 2 is 1
    //    When B=0 (non-mask): pass C (FP16 exponent 0x6400)
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(raw_05) : "r"(W05), "r"(mask_even_s3), "r"(magic_exp_s3));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(raw_16) : "r"(W16), "r"(mask_even_s3), "r"(magic_exp_s3));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(raw_27) : "r"(W27), "r"(mask_even_s3), "r"(magic_exp_s3));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(raw_38) : "r"(W38), "r"(mask_even_s3), "r"(magic_exp_s3));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(raw_49) : "r"(W49), "r"(mask_even_s3), "r"(magic_exp_s3));

    // 2. Hardware SIMD FP16 FMA: (raw * scale) + neg_bias_1028
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_05) : "r"(scale_32), "r"(neg_bias_1028_32));
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_16) : "r"(scale_32), "r"(neg_bias_1028_32));
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_27) : "r"(scale_32), "r"(neg_bias_1028_32));
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_38) : "r"(scale_32), "r"(neg_bias_1028_32));
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_49) : "r"(scale_32), "r"(neg_bias_1028_32));
}

// Pure PTX FP8 E4M3 -> FP16 Dequantization (0xFE bitwise OR + exp_bias addition + fma.rn.f16x2)
__device__ __forceinline__ void lop3_unpack_fp8_ptx(
    uint32_t W,
    uint32_t &raw_01, uint32_t &raw_23,
    uint32_t scale_32)
{
    const uint32_t exp_bias = 0x20002000; // Exponent offset +8 (0x8 << 10 = 0x2000) for lower and upper FP16

    // Unpack 4 FP8 bytes: W = [byte 3, byte 2, byte 1, byte 0]
    uint32_t W01 = W & 0xFFFF;
    uint32_t W23 = W >> 16;

    // Shift exponent/mantissa (bits 0..6) and sign (bit 7) into FP16 positions
    uint32_t W01_em   = ((W01 & 0x007F) << 7) | (((W01 >> 8) & 0x007F) << 23);
    uint32_t W01_sign = ((W01 & 0x0080) << 8) | (((W01 >> 8) & 0x0080) << 24);

    uint32_t W23_em   = ((W23 & 0x007F) << 7) | (((W23 >> 8) & 0x007F) << 23);
    uint32_t W23_sign = ((W23 & 0x0080) << 8) | (((W23 >> 8) & 0x0080) << 24);

    // Add exponent bias offset (+8) to mantissa/exponent word
    uint32_t W01_biased = W01_em + exp_bias;
    uint32_t W23_biased = W23_em + exp_bias;

    // Combine sign and biased exponent/mantissa
    raw_01 = W01_biased | W01_sign;
    raw_23 = W23_biased | W23_sign;

    // Optional scale adjustment via FMA (raw * scale + 0.0)
    const uint32_t zero_32 = 0x00000000;
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_01) : "r"(scale_32), "r"(zero_32));
    asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(raw_23) : "r"(scale_32), "r"(zero_32));
}
#endif // __CUDACC__

void launch_lop3_dequant_u4(
    const uint32_t* d_packed, half* d_output, const half* d_scale, const half* d_zero, int num_uint32s, cudaStream_t stream = 0);

void launch_lop3_dequant_s4(
    const uint32_t* d_packed, half* d_output, const half* d_scale, const half* d_zero, int num_uint32s, cudaStream_t stream = 0);

void launch_lop3_dequant_s3(
    const uint32_t* d_packed, half* d_output, const half* d_scale, const half* d_zero, int num_uint32s, cudaStream_t stream = 0);

void launch_lop3_dequant_fp8(
    const uint32_t* d_packed, half* d_output, const half* d_scale, int num_uint32s, cudaStream_t stream = 0);

#endif // LOP3_DEQUANT_H
