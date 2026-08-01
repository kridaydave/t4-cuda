#ifndef H17_MEGA_KERNEL_H
#define H17_MEGA_KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Launch function for Flagship H17 Fused INT3-Dequant Warp-Specialized GEMV Mega-Kernel
//
// Canonical H17 packing/layout contract (kernel + Python quantizer must match):
//   W_packed : (K/10) x N   int32, column-major words. Each uint32 packs 10 signed INT3
//              values along K for a single output column n (word index k_idx, column n at
//              W_packed[k_idx * N + n]). Bits [3*i .. 3*i+2] hold element i in [0..7],
//              i = 0..9; (q, zp, s) in quant units; kernel's magic-exponent path combines
//              sign-invert and offset so the enqueued zp/s are the plain quantizer values.
//   scale    : num_groups x N  FP16, group g covers K range [g*100, (g+1)*100).
//   zero_pt  : num_groups x N  FP16, same grouping. dequant: w = (q - zp_g) * s_g.
//   A        : M x K  FP16 row-major.   C : M x N FP16 row-major.
//   Quant group = 100 K-positions (10 packed words). num_groups = ceil(K / 100).
//   K must be a multiple of 10 for the uint32 packing; padding to a 100 boundary is
//   handled in the quantizer, and tail groups are masked by the K extent of A.
void launch_h17_gemv_s3(
    const half* A,
    const uint32_t* W_packed,
    const half* scale,
    const half* zero_point,
    half* C,
    int M, int N, int K,
    int num_groups,
    cudaStream_t stream = 0);

#endif // H17_MEGA_KERNEL_H
