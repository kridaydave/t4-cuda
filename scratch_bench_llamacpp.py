import subprocess
import sys
import os
import time
import threading

def run_cmd_streaming(cmd, check=True, log_file=None):
    print(f"Executing: {cmd}", flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"=== Command: {cmd} ===\n")
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    last_output_time = [time.time()]
    stop_heartbeat = threading.Event()

    def heartbeat():
        while not stop_heartbeat.is_set():
            time.sleep(5)
            if time.time() - last_output_time[0] >= 5:
                print(".", end="", flush=True)

    hb_thread = threading.Thread(target=heartbeat, daemon=True)
    hb_thread.start()
    
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        last_output_time[0] = time.time()
        print(line, end='', flush=True)
        output_lines.append(line)
        if log_file:
            with open(log_file, "a") as f:
                f.write(line)
                
    stop_heartbeat.set()
    hb_thread.join(timeout=1)
    process.stdout.close()
    return_code = process.wait()
    
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"=== Exit Code: {return_code} ===\n\n")
            
    if check and return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {cmd}")
        
    return "".join(output_lines)

def main():
    log_file = "/content/llamacpp_t4_cuda_int3_benchmark.log"
    summary_log = "/content/llamacpp_t4_cuda_int3_summary.log"
    
    with open(log_file, "w") as f:
        f.write(f"=== LLAMA.CPP CUDA INT3 BENCHMARK LOG ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
    with open(summary_log, "w") as f:
        f.write(f"=== LLAMA.CPP CUDA INT3 BENCHMARK SUMMARY ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 1. System Info & GPU verification
    run_cmd_streaming("nvidia-smi", log_file=log_file)

    # 2. Clone llama.cpp if needed
    if not os.path.exists("/content/llama.cpp"):
        run_cmd_streaming("git clone https://github.com/ggerganov/llama.cpp.git /content/llama.cpp", log_file=log_file)
    else:
        print("llama.cpp directory already exists.", flush=True)

    # 3. Build llama.cpp with CUDA support using Ninja for fast parallel build & live output streaming
    run_cmd_streaming("apt-get update -qq && apt-get install -y -qq ninja-build ccache", log_file=log_file)
    
    import shutil
    build_dir = "/content/llama.cpp/build"
    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir, exist_ok=True)
    
    cmake_cmd = "cmake -B /content/llama.cpp/build -S /content/llama.cpp -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA_FA=OFF -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_MMQ_ALL_QUANTS=OFF -DCMAKE_BUILD_TYPE=Release"
    run_cmd_streaming(cmake_cmd, log_file=log_file)
    
    build_targets_cmd = "cmake --build /content/llama.cpp/build --config Release --target llama-bench test-backend-ops"
    run_cmd_streaming(build_targets_cmd, log_file=log_file)
    
    # 4. Check available binaries
    run_cmd_streaming("ls -la /content/llama.cpp/build/bin", log_file=log_file)

    # 5. Run test-backend-ops for MUL_MAT / INT3
    print("Running ggml backend ops benchmarks for MUL_MAT...", flush=True)
    test_ops_bin = "/content/llama.cpp/build/bin/test-backend-ops"
    if os.path.exists(test_ops_bin):
        ops_output = run_cmd_streaming(f"{test_ops_bin} perf CUDA MUL_MAT", check=False, log_file=log_file)
        with open(summary_log, "a") as f:
            f.write("=== TEST BACKEND OPS PERF (MUL_MAT) ===\n")
            f.write(ops_output)
            f.write("\n")

    # 6. Benchmark llama-bench with synthetic matrix / quant types
    print("Running llama-bench benchmarks...", flush=True)
    bench_bin = "/content/llama.cpp/build/bin/llama-bench"
    if os.path.exists(bench_bin):
        bench_cmd = f"{bench_bin} -p 1,128,2048 -n 1 -b 1,4,16 -t 1 -o md"
        bench_output = run_cmd_streaming(bench_cmd, check=False, log_file=log_file)
        with open(summary_log, "a") as f:
            f.write("=== LLAMA-BENCH OUTPUT ===\n")
            f.write(bench_output)
            f.write("\n")

    print("Benchmark complete!", flush=True)
    print(f"Full log written to: {log_file}", flush=True)
    print(f"Summary log written to: {summary_log}", flush=True)

if __name__ == "__main__":
    main()
