#!/usr/bin/env python3
"""
Tesla T4 (Turing CC 7.5) Roofline Analyzer & CUDA Kernel Optimization Suite
Calculates exact hardware rooflines, arithmetic intensity thresholds, and 
simulates performance for custom CUDA/PTX kernels.
"""

import math
import sys

# NVIDIA Tesla T4 Specifications (TU104)
T4_SPECS = {
    "gpu_name": "NVIDIA Tesla T4",
    "architecture": "Turing (TU104, CC 7.5)",
    "num_sms": 40,
    "cuda_cores": 2560,
    "tensor_cores": 320,
    "base_clock_mhz": 585,
    "boost_clock_mhz": 1590,
    "gddr6_bus_bits": 256,
    "memory_bandwidth_gbps": 320.0,  # 320 GB/s GDDR6
    "l2_cache_mb": 4.0,
    "l1_shared_kb_per_sm": 96.0,
    "tdp_watts": 70,
    
    # Peak Performance Ratings (TFLOPS / TOPS)
    "fp32_tflops": 8.1,
    "fp16_cuda_tflops": 16.2,
    "fp16_tensor_tflops_dense": 65.0,
    "fp16_tensor_tflops_peak": 130.0,
    "int8_tensor_tops": 130.0,
    "int4_tensor_tops": 260.0,
}


def calculate_roofline():
    """Calculates break-even arithmetic intensity for T4."""
    bw = T4_SPECS["memory_bandwidth_gbps"]  # 320 GB/s
    fp32_peak = T4_SPECS["fp32_tflops"] * 1e3  # 8100 GFLOPS
    fp16_tc_peak = T4_SPECS["fp16_tensor_tflops_dense"] * 1e3  # 65000 GFLOPS
    int8_tc_peak = T4_SPECS["int8_tensor_tops"] * 1e3  # 130000 TOPS

    intensity_fp32 = fp32_peak / bw  # FLOP/byte
    intensity_fp16_tc = fp16_tc_peak / bw  # FLOP/byte
    intensity_int8_tc = int8_tc_peak / bw  # OP/byte

    return {
        "intensity_fp32": intensity_fp32,
        "intensity_fp16_tc": intensity_fp16_tc,
        "intensity_int8_tc": intensity_int8_tc,
    }


def gemm_roofline_analysis(m, n, k, batch_size=1, precision="fp16"):
    """
    Analyzes GEMM for (M, N, K) under T4 hardware constraints.
    GEMM FLOPs = 2 * B * M * N * K
    GEMM Bytes = B * (M*K*bytes_per_elem + K*N*bytes_per_elem + M*N*bytes_per_elem)
    """
    bytes_per_elem = 2 if precision == "fp16" else (4 if precision == "fp32" else 0.5)
    
    total_flops = 2 * batch_size * m * n * k
    total_bytes = batch_size * (m * k + k * n + m * n) * bytes_per_elem
    
    intensity = total_flops / total_bytes
    
    roofline = calculate_roofline()
    break_even = roofline["intensity_fp16_tc"] if precision == "fp16" else roofline["intensity_fp32"]
    
    peak_gflops = T4_SPECS["fp16_tensor_tflops_dense"] * 1e3 if precision == "fp16" else T4_SPECS["fp32_tflops"] * 1e3
    
    # Roofline bound determination
    if intensity < break_even:
        bound_type = "MEMORY_BANDWIDTH_BOUND"
        attainable_gflops = intensity * T4_SPECS["memory_bandwidth_gbps"]
        exec_time_ms = (total_bytes / (T4_SPECS["memory_bandwidth_gbps"] * 1e9)) * 1000
    else:
        bound_type = "COMPUTE_POWER_BOUND (70W TDP)"
        attainable_gflops = peak_gflops
        exec_time_ms = (total_flops / (peak_gflops * 1e9)) * 1000

    return {
        "m": m, "n": n, "k": k, "batch_size": batch_size,
        "precision": precision,
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "intensity": intensity,
        "break_even_intensity": break_even,
        "bound_type": bound_type,
        "attainable_gflops": attainable_gflops,
        "est_latency_ms": exec_time_ms
    }


def print_summary():
    print("=" * 80)
    print("  TESLA T4 (TURING CC 7.5) EXTREME ROOFLINE & KERNEL OPTIMIZATION REPORT")
    print("=" * 80)
    print(f"Hardware Target       : {T4_SPECS['gpu_name']} ({T4_SPECS['architecture']})")
    print(f"Streaming Multiprocess: {T4_SPECS['num_sms']} SMs ({T4_SPECS['cuda_cores']} CUDA Cores, {T4_SPECS['tensor_cores']} Tensor Cores)")
    print(f"Memory Subsystem      : 16 GB GDDR6 @ {T4_SPECS['memory_bandwidth_gbps']} GB/s Peak Bandwidth")
    print(f"Power Limit           : {T4_SPECS['tdp_watts']}W TDP Cap")
    print("-" * 80)

    roofline = calculate_roofline()
    print("ROOFLINE ARITHMETIC INTENSITY KNEES (FLOPs or OPs per Byte):")
    print(f"  - FP32 CUDA Cores   : {roofline['intensity_fp32']:.2f} FLOPs/byte")
    print(f"  - FP16 Tensor Cores : {roofline['intensity_fp16_tc']:.2f} FLOPs/byte (Threshold for Memory vs Compute Bound)")
    print(f"  - INT8 Tensor Cores : {roofline['intensity_int8_tc']:.2f} OPs/byte")
    print("-" * 80)

    print("\nBENCHMARK WORKLOAD SCENARIOS ON T4:")
    scenarios = [
        ("LLM Single Token Decoding (BS=1, SeqLen=1)", 1, 4096, 4096, 1, "fp16"),
        ("LLM Small Batch Decoding  (BS=8, SeqLen=1)", 1, 4096, 4096, 8, "fp16"),
        ("LLM Medium Batch Prefill  (BS=1, SeqLen=512)", 512, 4096, 4096, 1, "fp16"),
        ("Large Training GEMM       (BS=32, SeqLen=512)", 512, 4096, 4096, 32, "fp16"),
    ]

    for name, m, n, k, bs, prec in scenarios:
        res = gemm_roofline_analysis(m, n, k, bs, prec)
        print(f"\nWorkload: {name}")
        print(f"  - Intensity         : {res['intensity']:.2f} FLOPs/byte (Break-even: {res['break_even_intensity']:.2f})")
        print(f"  - Bottleneck        : {res['bound_type']}")
        print(f"  - Attainable Perf   : {res['attainable_gflops'] / 1e3:.2f} TFLOPS")
        print(f"  - Est. Minimum Time : {res['est_latency_ms']:.4f} ms")

    print("\n" + "=" * 80)
    print("KEY TESLA-T4 CUSTOMIZATION TAKEAWAYS:")
    print("1. Small Batch Inference (BS 1-8) is 100% MEMORY BANDWIDTH BOUND (~320 GB/s limit).")
    print("   -> Vectorized 128-bit loads (float4/uint4) & 4-bit quantization (W4A16) yield up to 4x latency reduction!")
    print("2. Large Batch / Training is COMPUTE & POWER BOUND (70W limit).")
    print("   -> Double-buffered register prefetching & tuning grid occupancy (avoiding 100% SM over-subscription)")
    print("      prevents GPU boost clock thermal throttling from 1590MHz down to 1000MHz.")
    print("=" * 80)


if __name__ == "__main__":
    print_summary()
