#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

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
// 1. Pointer-Chasing Latency Benchmark (L1, L2, DRAM)
// -------------------------------------------------------------------------
__global__ void latency_pointer_chase_kernel(const uint32_t * __restrict__ array, uint32_t *d_out, uint64_t *times, int iterations) {
    uint32_t current_idx = threadIdx.x;
    uint64_t start, end;
    
    // Warmup
    for (int i = 0; i < 100; i++) {
        current_idx = array[current_idx];
    }

    asm volatile("bar.sync 0;");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        // Dependent memory load forces serial memory access
        current_idx = array[current_idx];
    }
    
    asm volatile("bar.sync 0;");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    
    if (threadIdx.x == 0) {
        times[0] = end - start;
    }
    
    // Prevent compiler from eliminating memory loads
    if (threadIdx.x == 0) {
        d_out[0] = current_idx;
    }
}

// -------------------------------------------------------------------------
// 2. Shared Memory Bank Conflict Stall Cycles Benchmark
// -------------------------------------------------------------------------
__global__ void shmem_bank_conflict_kernel(uint32_t *out, uint64_t *times, int stride) {
    __shared__ uint32_t smem[1024];
    int tid = threadIdx.x;
    
    smem[tid] = tid;
    asm volatile("bar.sync 0;");
    
    uint64_t start, end;
    uint32_t val = tid;
    
    asm volatile("bar.sync 0;");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    
    #pragma unroll
    for (int i = 0; i < 1000; i++) {
        val ^= smem[(tid * stride + i) % 1024];
    }
    
    asm volatile("bar.sync 0;");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    
    if (tid == 0) {
        times[0] = end - start;
    }
    out[tid] = val;
}

// -------------------------------------------------------------------------
// 3. HMMA.884 Tensor Core Power & Boost Clock Throttling Benchmark
// -------------------------------------------------------------------------
__global__ void hmma_clock_decay_kernel(uint64_t *times, half *A, half *B, float *C, int loops) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
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

// Helper to construct a pointer-chasing linked list
void setup_pointer_chase(uint32_t *h_arr, size_t num_elements, size_t stride_elements) {
    for (size_t i = 0; i < num_elements; i++) {
        h_arr[i] = (uint32_t)((i + stride_elements) % num_elements);
    }
}

int main() {
    printf("===================================================\n");
    printf("Tesla T4 (Turing CC 7.5) Hardware Micro-Benchmarks\n");
    printf("===================================================\n");

    uint64_t *d_times;
    uint32_t *d_out;
    CHECK_CUDA(cudaMalloc(&d_times, 2 * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_out, 1024 * sizeof(uint32_t)));

    // ---------------------------------------------------------------------
    // Part 1: Shared Memory Bank Conflict Benchmark
    // ---------------------------------------------------------------------
    printf("\n--- [1] Shared Memory Bank Conflict Benchmark ---\n");
    int strides[4] = {1, 2, 4, 32};
    int conflict_labels[4] = {0, 2, 4, 32};
    int num_accesses = 1000;

    for (int i = 0; i < 4; i++) {
        uint64_t total_cycles = 0;
        shmem_bank_conflict_kernel<<<1, 32>>>(d_out, d_times, strides[i]);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&total_cycles, d_times, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        double cycles_per_access = (double)total_cycles / (double)num_accesses;
        printf("Stride %2d (%2d-way conflict): %6.1f total cycles (~%.2f cycles/access)\n", 
               strides[i], conflict_labels[i], (double)total_cycles, cycles_per_access);
    }

    // ---------------------------------------------------------------------
    // Part 2: Pointer-Chasing Memory Latency Benchmark (L1, L2, GDDR6 DRAM)
    // ---------------------------------------------------------------------
    printf("\n--- [2] Pointer-Chasing Memory Latency Benchmark ---\n");
    
    // L1 Cache Target (16 KB array < 64 KB L1)
    size_t l1_size = 4096; 
    uint32_t *h_l1 = (uint32_t*)malloc(l1_size * sizeof(uint32_t));
    setup_pointer_chase(h_l1, l1_size, 32);
    uint32_t *d_l1;
    CHECK_CUDA(cudaMalloc(&d_l1, l1_size * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(d_l1, h_l1, l1_size * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Warmup L1
    latency_pointer_chase_kernel<<<1, 32>>>(d_l1, d_out, d_times, 1000);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    uint64_t l1_cycles = 0;
    latency_pointer_chase_kernel<<<1, 32>>>(d_l1, d_out, d_times, 1000);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&l1_cycles, d_times, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    printf("L1 Data Cache Latency : %.1f cycles / load\n", (double)l1_cycles / 1000.0);

    // GDDR6 DRAM Target (64 MB array > 4 MB L2 Cache)
    size_t dram_size = 16 * 1024 * 1024; 
    uint32_t *h_dram = (uint32_t*)malloc(dram_size * sizeof(uint32_t));
    setup_pointer_chase(h_dram, dram_size, 1024);
    uint32_t *d_dram;
    CHECK_CUDA(cudaMalloc(&d_dram, dram_size * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(d_dram, h_dram, dram_size * sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint64_t dram_cycles = 0;
    latency_pointer_chase_kernel<<<1, 32>>>(d_dram, d_out, d_times, 100);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&dram_cycles, d_times, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    printf("GDDR6 DRAM Latency    : %.1f cycles / load\n", (double)dram_cycles / 100.0);

    free(h_l1);
    free(h_dram);
    cudaFree(d_l1);
    cudaFree(d_dram);

    // ---------------------------------------------------------------------
    // Part 3: Tensor Core HMMA Clock Throttling Profiling
    // ---------------------------------------------------------------------
    printf("\n--- [3] HMMA.884 Clock Throttling Profiling ---\n");
    half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, 256 * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, 256 * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, 256 * sizeof(float)));
    
    int loops = 10000000;
    
    // 100% Occupancy (1024 threads/SM)
    printf("Profiling 100%% Occupancy (1024 threads/SM)...\n");
    hmma_clock_decay_kernel<<<40, 1024>>>(d_times, d_A, d_B, d_C, loops);
    CHECK_CUDA(cudaDeviceSynchronize());
    uint64_t h_times[2];
    CHECK_CUDA(cudaMemcpy(h_times, d_times, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    printf("  -> 100%% Occupancy Duration: %lu cycles\n", h_times[1] - h_times[0]);
    
    // 25% Occupancy (256 threads/SM)
    printf("Profiling 25%% Occupancy (256 threads/SM)...\n");
    hmma_clock_decay_kernel<<<40, 256>>>(d_times, d_A, d_B, d_C, loops);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_times, d_times, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    printf("  -> 25%% Occupancy Duration : %lu cycles\n", h_times[1] - h_times[0]);

    cudaFree(d_times);
    cudaFree(d_out);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\n===================================================\n");
    printf("[BENCHMARK COMPLETE] Empirical Measurements Collected.\n");
    printf("===================================================\n");
    return 0;
}
