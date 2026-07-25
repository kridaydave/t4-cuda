#ifndef LOP3_DEQUANT_H
#define LOP3_DEQUANT_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __CUDACC__
// ============================================================================
// Unsigned INT4 Dequantization (u4 in [0..15])
// Uses magic exponent 0x6400 (1024.0 in FP16)
// ============================================================================
__device__ __forceinline__ void lop3_unpack_u4_to_4_half2(
    uint32_t W, 
    half2 &h2_04, half2 &h2_15, half2 &h2_26, half2 &h2_37,
    half2 scale_h2, half2 neg_bias_h2) 
{
    const uint32_t mask_even = 0x000F000F;
    const uint32_t magic_exp = 0x64006400; // 1024.0 in FP16
    
    uint32_t raw_04, raw_15, raw_26, raw_37;

    // LOP3 LUT 0xF2: (A & B) | C
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xF2;" : "=r"(raw_04) : "r"(W),       "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xF2;" : "=r"(raw_15) : "r"(W >> 4),  "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xF2;" : "=r"(raw_26) : "r"(W >> 8),  "r"(mask_even), "r"(magic_exp));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xF2;" : "=r"(raw_37) : "r"(W >> 12), "r"(mask_even), "r"(magic_exp));

    half2 &val_04 = reinterpret_cast<half2&>(raw_04);
    half2 &val_15 = reinterpret_cast<half2&>(raw_15);
    half2 &val_26 = reinterpret_cast<half2&>(raw_26);
    half2 &val_37 = reinterpret_cast<half2&>(raw_37);

    // neg_bias_h2 = (-1024.0 - zero_point) * scale
    h2_04 = __hfma2(val_04, scale_h2, neg_bias_h2);
    h2_15 = __hfma2(val_15, scale_h2, neg_bias_h2);
    h2_26 = __hfma2(val_26, scale_h2, neg_bias_h2);
    h2_37 = __hfma2(val_37, scale_h2, neg_bias_h2);
}

// ============================================================================
// Signed INT4 Two's Complement Dequantization (s4 in [-8..7])
// Uses LOP3 LUT 0x78 (bit 3 sign inversion) and magic offset 1032.0 (0x6408)
// ============================================================================
__device__ __forceinline__ void lop3_unpack_s4_to_4_half2(
    uint32_t W, 
    half2 &h2_04, half2 &h2_15, half2 &h2_26, half2 &h2_37,
    half2 scale_h2, half2 neg_bias_1032_h2) 
{
    const uint32_t mask_even     = 0x000F000F;
    const uint32_t magic_exp_s4 = 0x64006400; // Exponent 1024.0 FP16

    uint32_t raw_04, raw_15, raw_26, raw_37;

    // LOP3 LUT 0x78: Inverts Bit 3 (sign bit of INT4) while masking & ORing 0x6400
    // Result: (1032.0 + s4) in FP16 for all s4 in [-8..7]
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x78;" : "=r"(raw_04) : "r"(W),       "r"(mask_even), "r"(magic_exp_s4));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x78;" : "=r"(raw_15) : "r"(W >> 4),  "r"(mask_even), "r"(magic_exp_s4));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x78;" : "=r"(raw_26) : "r"(W >> 8),  "r"(mask_even), "r"(magic_exp_s4));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x78;" : "=r"(raw_37) : "r"(W >> 12), "r"(mask_even), "r"(magic_exp_s4));

    half2 &val_04 = reinterpret_cast<half2&>(raw_04);
    half2 &val_15 = reinterpret_cast<half2&>(raw_15);
    half2 &val_26 = reinterpret_cast<half2&>(raw_26);
    half2 &val_37 = reinterpret_cast<half2&>(raw_37);

    // neg_bias_1032_h2 = (-1032.0 - zero_point) * scale
    h2_04 = __hfma2(val_04, scale_h2, neg_bias_1032_h2);
    h2_15 = __hfma2(val_15, scale_h2, neg_bias_1032_h2);
    h2_26 = __hfma2(val_26, scale_h2, neg_bias_1032_h2);
    h2_37 = __hfma2(val_37, scale_h2, neg_bias_1032_h2);
}
#endif // __CUDACC__

void launch_lop3_dequant_u4(
    const uint32_t* d_packed, half* d_output, const half* d_scale, const half* d_zero, int num_uint32s, cudaStream_t stream = 0);

void launch_lop3_dequant_s4(
    const uint32_t* d_packed, half* d_output, const half* d_scale, const half* d_zero, int num_uint32s, cudaStream_t stream = 0);

#endif // LOP3_DEQUANT_H
