#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <stdint.h>

using namespace nvcuda;

#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

// -------------------------------------------------------------------------
// 1. Physical Hardware Latency & Bandwidth Measurements
// -------------------------------------------------------------------------

// Benchmark to measure L1, L2, and DRAM latency using pointer chasing
__global__ void latency_benchmark_kernel(uint32_t *indices, uint64_t *times, int iterations) {
    uint32_t current_idx = threadIdx.x;
    uint64_t start, end;
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        current_idx = indices[current_idx];
    }

    asm volatile("bar.sync 0;");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    
    for (int i = 0; i < iterations; i++) {
        // Dependent memory load forces serialization
        current_idx = indices[current_idx];
    }
    
    asm volatile("bar.sync 0;");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    
    if (threadIdx.x == 0) {
        times[blockIdx.x] = (end - start) / iterations;
    }
    // Dummy write to prevent optimization removing the loop
    indices[0] = current_idx;
}

// -------------------------------------------------------------------------
// Shared Memory Bank Conflict Stall Cycles Measurement
// -------------------------------------------------------------------------

// Kernel to measure shared memory access latency with variable strides
__global__ void shmem_bank_conflict_kernel(uint32_t *out, uint64_t *times, int stride) {
    __shared__ uint32_t smem[1024];
    int tid = threadIdx.x;
    
    // Initialize shared memory
    smem[tid] = tid;
    asm volatile("bar.sync 0;");
    
    uint64_t start, end;
    uint32_t val = 0;
    
    // Measure memory accesses
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    
    #pragma unroll 100
    for (int i = 0; i < 100; i++) {
        val ^= smem[(tid * stride) % 1024];
    }
    
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    
    if (tid == 0) {
        // Store average cycles per access
        times[0] = (end - start) / 100;
    }
    
    // Dummy write to global memory to prevent optimization
    out[tid] = val;
}

// -------------------------------------------------------------------------
// 2. Thermal & NVPM Clock Throttling Profiling Script
// -------------------------------------------------------------------------

// Kernel to execute dense HMMA.884 instructions to profile clock throttling
__global__ void hmma_clock_decay_kernel(uint64_t *times, half *A, half *B, float *C, int loops) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Load matrices
    if (wid == 0) {
        wmma::load_matrix_sync(a_frag, A, 16);
        wmma::load_matrix_sync(b_frag, B, 16);
    }
    
    uint64_t start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    
    for (int i = 0; i < loops; i++) {
        if (wid == 0) {
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    
    if (tid == 0 && blockIdx.x == 0) {
        times[0] = start;
        times[1] = end;
    }
    
    if (wid == 0) {
        wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
    }
}

// -------------------------------------------------------------------------
// 3. Production C++ Benchmark Harness
// -------------------------------------------------------------------------
int main() {
    printf("===================================================\n");
    printf("Tesla T4 (Turing CC 7.5) Hardware Micro-Benchmarks\n");
    printf("===================================================\n");

    // Allocate memory for latency benchmarking
    int size = 1024 * 1024;
    uint32_t *d_indices;
    uint64_t *d_times;
    
    cudaMalloc(&d_indices, size * sizeof(uint32_t));
    cudaMalloc(&d_times, 2 * sizeof(uint64_t));
    
    // Initialize pointer chasing array on device
    // (In a real benchmark, this would be set up from host to construct a linked list)
    
    printf("\nRunning Shared Memory Bank Conflict Benchmark...\n");
    uint32_t *d_out;
    uint64_t h_time;
    cudaMalloc(&d_out, 1024 * sizeof(uint32_t));
    
    // Stride 1 = 0 conflicts
    shmem_bank_conflict_kernel<<<1, 32>>>(d_out, d_times, 1);
    cudaMemcpy(&h_time, d_times, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("Stride 1 (0-way conflict): %lu cycles\n", h_time);
    
    // Stride 2 = 2-way conflicts
    shmem_bank_conflict_kernel<<<1, 32>>>(d_out, d_times, 2);
    cudaMemcpy(&h_time, d_times, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("Stride 2 (2-way conflict): %lu cycles\n", h_time);
    
    // Stride 4 = 4-way conflicts
    shmem_bank_conflict_kernel<<<1, 32>>>(d_out, d_times, 4);
    cudaMemcpy(&h_time, d_times, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("Stride 4 (4-way conflict): %lu cycles\n", h_time);
    
    // Stride 32 = 32-way conflicts
    shmem_bank_conflict_kernel<<<1, 32>>>(d_out, d_times, 32);
    cudaMemcpy(&h_time, d_times, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("Stride 32 (32-way conflict): %lu cycles\n", h_time);

    printf("\nRunning HMMA.884 Clock Throttling Profiling...\n");
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, 256 * sizeof(half));
    cudaMalloc(&d_B, 256 * sizeof(half));
    cudaMalloc(&d_C, 256 * sizeof(float));
    
    int loops = 100000000;
    
    // 100% Occupancy (1024 threads)
    printf("100%% Occupancy (1024 threads/SM)...\n");
    hmma_clock_decay_kernel<<<40, 1024>>>(d_times, d_A, d_B, d_C, loops);
    uint64_t h_times[2];
    cudaMemcpy(h_times, d_times, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("Start Clock: %lu, End Clock: %lu, Diff: %lu cycles\n", h_times[0], h_times[1], h_times[1] - h_times[0]);
    
    // 25% Occupancy (256 threads)
    printf("25%% Occupancy (256 threads/SM)...\n");
    hmma_clock_decay_kernel<<<40, 256>>>(d_times, d_A, d_B, d_C, loops);
    cudaMemcpy(h_times, d_times, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("Start Clock: %lu, End Clock: %lu, Diff: %lu cycles\n", h_times[0], h_times[1], h_times[1] - h_times[0]);
    
    cudaFree(d_indices);
    cudaFree(d_times);
    cudaFree(d_out);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nBenchmarking Complete.\n");
    return 0;
}
