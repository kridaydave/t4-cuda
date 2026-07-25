#ifndef T4_PTX_ASSEMBLY_SUITE_CU
#define T4_PTX_ASSEMBLY_SUITE_CU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ---------------------------------------------------------
// 1. LOP3 Operations
// ---------------------------------------------------------



__device__ __forceinline__ uint32_t lop3_0x64(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x64;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ uint32_t lop3_0xE2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xE2;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ uint32_t lop3_0xF2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xF2;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ uint32_t lop3_0xEA(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ uint32_t lop3_0x6A(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// ---------------------------------------------------------
// 2. LDMATRIX
// ---------------------------------------------------------

__device__ __forceinline__ void ldmatrix_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void* smem_ptr) {
    uint32_t smem_addr;
    asm volatile("{ .reg .u64 u64addr;\n"
                 " cvta.to.shared.u64 u64addr, %1;\n"
                 " cvt.u32.u64 %0, u64addr; }\n"
                 : "=r"(smem_addr) : "l"(smem_ptr));
                 
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b64 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void* smem_ptr) {
    uint32_t smem_addr;
    asm volatile("{ .reg .u64 u64addr;\n"
                 " cvta.to.shared.u64 u64addr, %1;\n"
                 " cvt.u32.u64 %0, u64addr; }\n"
                 : "=r"(smem_addr) : "l"(smem_ptr));
                 
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b64 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr)
    );
}

// ---------------------------------------------------------
// 3. MMA.SYNC
// ---------------------------------------------------------

__device__ __forceinline__ void mma_m16n8k8_fp16_fp32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1,
    uint32_t b0,
    float c0, float c1, float c2, float c3) 
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1),
          "r"(b0),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

// ---------------------------------------------------------
// 4. Warp Shuffles
// ---------------------------------------------------------

__device__ __forceinline__ float shfl_sync_idx(float var, int srcLane, int width=32) {
    float ret;
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, %3;"
                 : "=f"(ret) : "f"(var), "r"(srcLane), "r"(0x1f00 | (width-1)));
    return ret;
}

__device__ __forceinline__ float shfl_sync_bfly(float var, int laneMask, int width=32) {
    float ret;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, %3;"
                 : "=f"(ret) : "f"(var), "r"(laneMask), "r"(0x1f00 | (width-1)));
    return ret;
}

#endif // T4_PTX_ASSEMBLY_SUITE_CU
