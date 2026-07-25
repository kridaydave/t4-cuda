#ifndef FUSED_W4A16_GEMM_H
#define FUSED_W4A16_GEMM_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

/**
 * Fused Unsigned INT4 (W4A16) GEMM / GEMV Kernel
 * 
 * Performs on-the-fly LOP3 0xEA dequantization inside GPU registers:
 *   C = (Dequant(W_u4) * Scale - Bias) * A
 * 
 * @param A Pointer to activation matrix (M x K) in FP16 row-major
 * @param W_packed Pointer to packed 4-bit weights (K/8 x N) as uint32_t
 * @param scale Pointer to per-column scales (1 x N) in FP16
 * @param zero_point Pointer to per-column zero-points (1 x N) in FP16
 * @param C Pointer to output matrix (M x N) in FP16
 * @param M Row count of activations / output
 * @param N Column count of weights / output
 * @param K Inner dimension
 * @param stream CUDA stream
 */
void launch_fused_w4a16_gemm_u4(
    const half* d_A,
    const uint32_t* d_W_packed,
    const half* d_scale,
    const half* d_zero,
    half* d_C,
    int M, int N, int K,
    cudaStream_t stream = 0);

/**
 * Fused Signed Two's Complement INT4 (S4A16) GEMM / GEMV Kernel
 * 
 * Performs on-the-fly single-cycle LOP3 0x6A sign-flip + exponent injection in registers:
 *   C = (Dequant(W_s4) * Scale - Bias_1032) * A
 */
void launch_fused_w4a16_gemm_s4(
    const half* d_A,
    const uint32_t* d_W_packed,
    const half* d_scale,
    const half* d_zero,
    half* d_C,
    int M, int N, int K,
    cudaStream_t stream = 0);

#endif // FUSED_W4A16_GEMM_H
