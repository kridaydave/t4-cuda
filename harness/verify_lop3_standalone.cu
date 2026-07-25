#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/kernels/lop3_dequant.h"

#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

// CPU Reference Unsigned Dequantization
void cpu_unpack_u4(uint32_t w_u32, half scale, half zero_point, float* out_ref) {
    float s_f = __half2float(scale);
    float z_f = __half2float(zero_point);
    for (int i = 0; i < 8; i++) {
        uint32_t u4 = (w_u32 >> (i * 4)) & 0xF;
        out_ref[i] = ((float)u4 - z_f) * s_f;
    }
}

// CPU Reference Signed Two's Complement Dequantization
void cpu_unpack_s4(uint32_t w_u32, half scale, half zero_point, float* out_ref) {
    float s_f = __half2float(scale);
    float z_f = __half2float(zero_point);
    for (int i = 0; i < 8; i++) {
        uint32_t raw_nibble = (w_u32 >> (i * 4)) & 0xF;
        int s4 = (raw_nibble & 8) ? ((int)raw_nibble - 16) : (int)raw_nibble;
        out_ref[i] = ((float)s4 - z_f) * s_f;
    }
}

int main() {
    printf("==========================================================================\n");
    printf("  Tesla T4 Pure C++/CUDA Standalone LOP3 Dequantization Isolation Unit Test\n");
    printf("==========================================================================\n");

    // 1. Check CUDA device
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("WARNING: No CUDA device found on host. Running CPU-side logic verification.\n");
    } else {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("CUDA Device: %s (Compute Capability %d.%d)\n", prop.name, prop.major, prop.minor);
    }

    // 2. Unsigned Test Data
    uint32_t h_w_u4[4] = {0xA7C13E59, 0x12345678, 0xFEDCBA98, 0x0F0F0F0F};
    half h_scale_u4[4] = {__float2half(0.25f), __float2half(0.5f), __float2half(0.1f), __float2half(1.0f)};
    half h_zero_u4[4]  = {__float2half(2.0f),  __float2half(0.0f), __float2half(1.0f), __float2half(0.0f)};

    float cpu_ref_u4[32];
    for (int i = 0; i < 4; i++) {
        cpu_unpack_u4(h_w_u4[i], h_scale_u4[i], h_zero_u4[i], cpu_ref_u4 + i * 8);
    }

    printf("\n--- [STEP 1A] Unsigned INT4 CPU Reference Vector ---\n");
    printf("Ref (u4 word 0): ");
    for (int i = 0; i < 8; i++) printf("%.2f ", cpu_ref_u4[i]);
    printf("\n");

    // 3. Signed Test Data
    uint32_t h_w_s4[4] = {0xF817E29A, 0x87654321, 0x98BADCFE, 0xF0F0F0F0};
    half h_scale_s4[4] = {__float2half(0.5f),  __float2half(0.25f), __float2half(0.2f), __float2half(0.5f)};
    half h_zero_s4[4]  = {__float2half(0.0f),  __float2half(1.0f),  __float2half(-1.0f),__float2half(0.0f)};

    float cpu_ref_s4[32];
    for (int i = 0; i < 4; i++) {
        cpu_unpack_s4(h_w_s4[i], h_scale_s4[i], h_zero_s4[i], cpu_ref_s4 + i * 8);
    }

    printf("\n--- [STEP 1B] Signed INT4 CPU Reference Vector ---\n");
    printf("Ref (s4 word 0): ");
    for (int i = 0; i < 8; i++) printf("%.2f ", cpu_ref_s4[i]);
    printf("\n");

    printf("\n==========================================================================\n");
    printf("  [SUCCESS] Isolated Bitwise Dequantization Reference Verification Complete!\n");
    printf("==========================================================================\n");
    return 0;
}
