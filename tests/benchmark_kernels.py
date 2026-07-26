import torch
import time
import sys

try:
    import t4_kernels
    HAS_T4 = True
except ImportError:
    HAS_T4 = False
    print("WARNING: t4_kernels not found.")

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

def benchmark_latency(func, *args, warmup=100, iters=1000):
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    
    for i in range(iters):
        start_events[i].record()
        func(*args)
        end_events[i].record()
        
    torch.cuda.synchronize()
    
    times_us = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
    times_us.sort()
    
    return {
        'median': times_us[iters // 2],
        'mean': sum(times_us) / iters,
        'p5': times_us[int(iters * 0.05)],
        'p95': times_us[int(iters * 0.95)]
    }

def print_row(name, stats, memory_mb=None, bandwidth_gb=None, speedup=None):
    mem_str = f"{memory_mb:.1f} MB" if memory_mb is not None else "-"
    bw_str = f"{bandwidth_gb:.1f} GB/s" if bandwidth_gb is not None else "-"
    su_str = f"{speedup:.2f}x" if speedup is not None else "-"
    print(f"{name:<25} | {stats['median']:>8.2f} us | {mem_str:>10} | {bw_str:>10} | {su_str:>8}")

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot benchmark.")
        return
        
    print("=====================================================================================")
    print(f"{'Benchmark':<25} | {'Latency':>11} | {'Peak VRAM':>10} | {'Bandwidth':>10} | {'Speedup':>8}")
    print("=====================================================================================")
    
    # 1. Dequant Throughput
    sizes = [1024, 64*1024, 1024*1024, 16*1024*1024]
    for size in sizes:
        if not HAS_T4: continue
        packed = torch.randint(0, 2**31-1, (size,), dtype=torch.int32, device='cuda')
        scales = torch.ones(1, size*8, dtype=torch.float16, device='cuda')
        zeros = torch.zeros(1, size*8, dtype=torch.float16, device='cuda')
        stats = benchmark_latency(t4_kernels.dequantize_u4, packed, scales, zeros)
        bytes_read = size * 4 + size * 8 * 2 * 2
        bytes_written = size * 8 * 2
        bandwidth = (bytes_read + bytes_written) / (stats['median'] * 1e-6) / 1e9
        print_row(f"Dequant {size} packed", stats, bandwidth_gb=bandwidth)
        
    # 2. Fused GEMM Latency
    shapes = [
        (1, 3584, 3584, "Attn Proj"),
        (1, 3584, 9216, "MLP Up"),
        (1, 9216, 3584, "MLP Down"),
        (1, 3584, 151936, "Vocab Proj")
    ]
    
    for M, K, N, name in shapes:
        A = torch.randn(M, K, dtype=torch.float16, device='cuda')
        W_fp16 = torch.randn(K, N, dtype=torch.float16, device='cuda')
        W_packed = torch.randint(0, 2**31-1, (K//8, N), dtype=torch.int32, device='cuda')
        scales = torch.ones(1, N, dtype=torch.float16, device='cuda')
        zeros = torch.zeros(1, N, dtype=torch.float16, device='cuda')
        
        torch.cuda.reset_peak_memory_stats()
        base_stats = benchmark_latency(torch.nn.functional.linear, A, W_fp16.t())
        base_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print_row(f"Base {name}", base_stats, memory_mb=base_mem)
        
        if HAS_T4:
            torch.cuda.reset_peak_memory_stats()
            fused_stats = benchmark_latency(t4_kernels.fused_w4a16_gemm_u4, A, W_packed, scales, zeros)
            fused_mem = torch.cuda.max_memory_allocated() / (1024**2)
            speedup = base_stats['median'] / fused_stats['median']
            print_row(f"Fused {name}", fused_stats, memory_mb=fused_mem, speedup=speedup)
            
        if HAS_BNB:
            # Fake BNB benchmark for structural comparison
            pass

    print("\nSummary against Targets:")
    print("- H4 Target (2.50x SASS speedup): Check 'Speedup' column vs Base.")
    print("- H7 Target (3.08x speedup, 303 GB/s): Check Dequant throughput.")
    print("- H8 Target (94.2% fetch stall reduction): Indirectly measured by overall latency.")

if __name__ == '__main__':
    main()
