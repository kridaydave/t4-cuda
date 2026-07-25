#include <torch/extension.h>
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

    launch_lop3_dequant_u4(
        reinterpret_cast<const uint32_t*>(packed_weights.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_points.data_ptr<at::Half>()),
        num_uint32s,
        at::cuda::getCurrentCUDAStream());

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

    launch_lop3_dequant_s4(
        reinterpret_cast<const uint32_t*>(packed_weights.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_points.data_ptr<at::Half>()),
        num_uint32s,
        at::cuda::getCurrentCUDAStream());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize_u4", &dequantize_lop3_u4_cuda, "LOP3 Fast Unsigned INT4 Dequantization (CUDA)");
    m.def("dequantize_s4", &dequantize_lop3_s4_cuda, "LOP3 Fast Signed INT4 Dequantization (CUDA)");
}
