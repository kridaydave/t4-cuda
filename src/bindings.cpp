#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include "kernels/lop3_dequant.h"
#include "kernels/h17_mega_kernel.h"

torch::Tensor dequantize_lop3_u4_cuda(
    torch::Tensor packed_weights,
    torch::Tensor scales,
    torch::Tensor zero_points)
{
    TORCH_CHECK(packed_weights.is_cuda(), "packed_weights must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(zero_points.is_cuda(), "zero_points must be a CUDA tensor");
    TORCH_CHECK(packed_weights.scalar_type() == torch::kInt32, "packed_weights must be Int32");

    int num_uint32s = packed_weights.numel();
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(packed_weights.device());
    torch::Tensor output = torch::empty({num_uint32s * 8}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_lop3_dequant_u4(
        reinterpret_cast<const uint32_t*>(packed_weights.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_points.data_ptr<at::Half>()),
        num_uint32s,
        stream);

    return output;
}

torch::Tensor dequantize_lop3_s4_cuda(
    torch::Tensor packed_weights,
    torch::Tensor scales,
    torch::Tensor zero_points)
{
    TORCH_CHECK(packed_weights.is_cuda(), "packed_weights must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(zero_points.is_cuda(), "zero_points must be a CUDA tensor");
    TORCH_CHECK(packed_weights.scalar_type() == torch::kInt32, "packed_weights must be Int32");

    int num_uint32s = packed_weights.numel();
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(packed_weights.device());
    torch::Tensor output = torch::empty({num_uint32s * 8}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_lop3_dequant_s4(
        reinterpret_cast<const uint32_t*>(packed_weights.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_points.data_ptr<at::Half>()),
        num_uint32s,
        stream);

    return output;
}

#include "kernels/fused_w4a16_gemm.h"

torch::Tensor fused_w4a16_gemm_u4_cuda(
    torch::Tensor A,
    torch::Tensor W_packed,
    torch::Tensor scales,
    torch::Tensor zero_points)
{
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(W_packed.is_cuda(), "W_packed must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(zero_points.is_cuda(), "zero_points must be a CUDA tensor");

    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be FP16");
    TORCH_CHECK(W_packed.scalar_type() == torch::kInt32, "W_packed must be Int32");

    int M = A.size(0);
    int K = A.size(1);
    int N = W_packed.size(1);

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_fused_w4a16_gemm_u4(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_points.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K,
        stream);

    return C;
}

torch::Tensor fused_w4a16_gemm_s4_cuda(
    torch::Tensor A,
    torch::Tensor W_packed,
    torch::Tensor scales,
    torch::Tensor zero_points)
{
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(W_packed.is_cuda(), "W_packed must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(zero_points.is_cuda(), "zero_points must be a CUDA tensor");

    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be FP16");
    TORCH_CHECK(W_packed.scalar_type() == torch::kInt32, "W_packed must be Int32");

    int M = A.size(0);
    int K = A.size(1);
    int N = W_packed.size(1);

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_fused_w4a16_gemm_s4(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_points.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K,
        stream);

    return C;
}

torch::Tensor dequantize_lop3_s3_cuda(
    torch::Tensor packed_weights,
    torch::Tensor scales,
    torch::Tensor zero_points)
{
    TORCH_CHECK(packed_weights.is_cuda(), "packed_weights must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(zero_points.is_cuda(), "zero_points must be a CUDA tensor");
    TORCH_CHECK(packed_weights.scalar_type() == torch::kInt32, "packed_weights must be Int32");

    int num_uint32s = packed_weights.numel();
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(packed_weights.device());
    torch::Tensor output = torch::empty({num_uint32s * 10}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_lop3_dequant_s3(
        reinterpret_cast<const uint32_t*>(packed_weights.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_points.data_ptr<at::Half>()),
        num_uint32s,
        stream);

    return output;
}

torch::Tensor dequantize_lop3_fp8_cuda(
    torch::Tensor packed_weights,
    torch::Tensor scales)
{
    TORCH_CHECK(packed_weights.is_cuda(), "packed_weights must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(packed_weights.scalar_type() == torch::kInt32, "packed_weights must be Int32");

    int num_uint32s = packed_weights.numel();
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(packed_weights.device());
    torch::Tensor output = torch::empty({num_uint32s * 4}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_lop3_dequant_fp8(
        reinterpret_cast<const uint32_t*>(packed_weights.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        num_uint32s,
        stream);

    return output;
}

torch::Tensor fused_h17_gemv_s3_cuda(
    torch::Tensor A,
    torch::Tensor W_packed,
    torch::Tensor scales,
    torch::Tensor zero_points)
{
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(W_packed.is_cuda(), "W_packed must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    if (zero_points.defined() && zero_points.numel() > 0) {
        TORCH_CHECK(zero_points.is_cuda(), "zero_points must be a CUDA tensor");
        TORCH_CHECK(zero_points.scalar_type() == torch::kHalf, "zero_points must be FP16");
        TORCH_CHECK(zero_points.is_contiguous(), "zero_points must be contiguous");
    }

    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(W_packed.is_contiguous(), "W_packed must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");

    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be FP16");
    TORCH_CHECK(W_packed.scalar_type() == torch::kInt32, "W_packed must be Int32");
    TORCH_CHECK(scales.scalar_type() == torch::kHalf, "scales must be FP16");

    TORCH_CHECK(A.dim() == 1 || A.dim() == 2, "A must be 1D or 2D tensor");
    TORCH_CHECK(W_packed.dim() == 2, "W_packed must be a 2D tensor");

    int M = A.dim() == 1 ? 1 : A.size(0);
    int K = A.dim() == 1 ? A.size(0) : A.size(1);
    int N = W_packed.size(1);

    TORCH_CHECK(M > 0 && N > 0 && K > 0, "Dimensions M, N, K must be positive");
    // Canonical H17 packing: 10 INT3 per uint32, W_packed is (K/10) x N.
    TORCH_CHECK(K % 10 == 0, "K must be a multiple of 10 for H17 int3 packing");
    TORCH_CHECK(W_packed.size(0) == K / 10,
                "W_packed first dim must equal K/10 (canonical H17 layout)");

    // Quant group = 100 K-positions; scales/zps are per-group.
    const int num_groups = (K + 99) / 100;
    TORCH_CHECK(scales.numel() == (int64_t)num_groups * N,
                "scales must have num_groups * N elements (num_groups = ceil(K/100))");
    if (zero_points.defined() && zero_points.numel() > 0) {
        TORCH_CHECK(zero_points.numel() == (int64_t)num_groups * N,
                    "zero_points must have num_groups * N elements");
    }

    at::cuda::CUDAGuard device_guard(A.device());

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const half* zp_ptr = zero_points.defined() && zero_points.numel() > 0 ?
                         reinterpret_cast<const half*>(zero_points.data_ptr<at::Half>()) : nullptr;

    launch_h17_gemv_s3(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        zp_ptr,
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, num_groups,
        stream);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize_u4", &dequantize_lop3_u4_cuda, "LOP3 Fast Unsigned INT4 Dequantization (CUDA)");
    m.def("dequantize_s4", &dequantize_lop3_s4_cuda, "LOP3 Fast Signed INT4 Dequantization (CUDA)");
    m.def("dequantize_s3", &dequantize_lop3_s3_cuda, "LOP3 Fast Signed INT3 Dequantization (CUDA)");
    m.def("dequantize_fp8", &dequantize_lop3_fp8_cuda, "LOP3 Fast FP8 E4M3 Dequantization (CUDA)");
    m.def("fused_w4a16_gemm_u4", &fused_w4a16_gemm_u4_cuda, "Fused Unsigned W4A16 GEMM with LOP3 0xEA Dequant (CUDA)");
    m.def("fused_w4a16_gemm_s4", &fused_w4a16_gemm_s4_cuda, "Fused Signed S4A16 GEMM with LOP3 0x6A Dequant (CUDA)");
    m.def("fused_h17_gemv_s3", &fused_h17_gemv_s3_cuda, "Flagship H17 Fused INT3 Dequant + GEMV Decode Mega-Kernel (CUDA)");
}
