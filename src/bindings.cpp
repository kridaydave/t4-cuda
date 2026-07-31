#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include "kernels/lop3_dequant.h"

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize_u4", &dequantize_lop3_u4_cuda, "LOP3 Fast Unsigned INT4 Dequantization (CUDA)");
    m.def("dequantize_s4", &dequantize_lop3_s4_cuda, "LOP3 Fast Signed INT4 Dequantization (CUDA)");
    m.def("dequantize_s3", &dequantize_lop3_s3_cuda, "LOP3 Fast Signed INT3 Dequantization (CUDA)");
    m.def("dequantize_fp8", &dequantize_lop3_fp8_cuda, "LOP3 Fast FP8 E4M3 Dequantization (CUDA)");
    m.def("fused_w4a16_gemm_u4", &fused_w4a16_gemm_u4_cuda, "Fused Unsigned W4A16 GEMM with LOP3 0xEA Dequant (CUDA)");
    m.def("fused_w4a16_gemm_s4", &fused_w4a16_gemm_s4_cuda, "Fused Signed S4A16 GEMM with LOP3 0x6A Dequant (CUDA)");
}
