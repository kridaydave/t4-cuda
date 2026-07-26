#!/usr/bin/env python3
"""
run_all_cuda_tests.py — Master test runner for Epoch-1 CUDA kernel verification.

Usage:
    python research/tests/run_all_cuda_tests.py [--quick] [--bench] [--all]

Flags:
    --quick   Run only correctness tests (skip benchmarks)
    --bench   Run only benchmarks (skip correctness)
    --all     Run everything (default)

Requires: t4_kernels extension built via `cd research/src && pip install -e .`
"""

import sys
import os
import subprocess
import time
import argparse
from pathlib import Path

TESTS_DIR = Path(__file__).parent
RESEARCH_DIR = TESTS_DIR.parent
SRC_DIR = RESEARCH_DIR / "src"

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(title: str):
    print(f"\n{CYAN}{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}{RESET}\n")


def print_result(name: str, passed: bool, duration: float):
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  [{status}] {name} ({duration:.1f}s)")


def check_gpu():
    """Check if a CUDA GPU is available."""
    print_header("GPU DETECTION")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                print(f"  GPU: {line.strip()}")
            return True
        else:
            print(f"  {RED}nvidia-smi failed: {result.stderr.strip()}{RESET}")
            return False
    except FileNotFoundError:
        print(f"  {RED}nvidia-smi not found — no NVIDIA GPU detected{RESET}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  {RED}nvidia-smi timed out{RESET}")
        return False


def check_kernels_built():
    """Check if t4_kernels extension is installed."""
    try:
        import t4_kernels
        print(f"  {GREEN}t4_kernels extension found{RESET}")
        funcs = ['dequantize_u4', 'dequantize_s4',
                 'fused_w4a16_gemm_u4', 'fused_w4a16_gemm_s4']
        for f in funcs:
            if hasattr(t4_kernels, f):
                print(f"    ✓ {f}")
            else:
                print(f"    {RED}✗ {f} MISSING{RESET}")
                return False
        return True
    except ImportError:
        print(f"  {YELLOW}t4_kernels not installed. Building...{RESET}")
        return build_kernels()


def build_kernels():
    """Build the t4_kernels extension."""
    print(f"  Running: pip install -e {SRC_DIR}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(SRC_DIR)],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode == 0:
        print(f"  {GREEN}Build successful{RESET}")
        return True
    else:
        print(f"  {RED}Build failed:{RESET}")
        # Print last 20 lines of error
        stderr_lines = result.stderr.strip().split("\n")
        for line in stderr_lines[-20:]:
            print(f"    {line}")
        return False


def run_test_script(script_path: str, label: str) -> tuple[bool, float]:
    """Run a test script and return (passed, duration_seconds)."""
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, timeout=600,
        cwd=str(TESTS_DIR)
    )
    duration = time.time() - start

    # Print output
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")

    if result.returncode != 0:
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-10:]:
                print(f"    {RED}{line}{RESET}")

    return result.returncode == 0, duration


def main():
    parser = argparse.ArgumentParser(description="Epoch-1 CUDA Kernel Test Suite")
    parser.add_argument("--quick", action="store_true",
                        help="Correctness tests only (skip benchmarks)")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmarks only (skip correctness)")
    parser.add_argument("--all", action="store_true", default=True,
                        help="Run everything (default)")
    args = parser.parse_args()

    run_correctness = not args.bench
    run_benchmarks = not args.quick

    print_header("EPOCH-1 CUDA KERNEL VERIFICATION SUITE")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'Correctness + Benchmarks' if run_correctness and run_benchmarks else 'Correctness Only' if run_correctness else 'Benchmarks Only'}")

    # Step 1: GPU check
    has_gpu = check_gpu()
    if not has_gpu:
        print(f"\n  {RED}No GPU detected. These tests require an NVIDIA GPU (ideally T4).{RESET}")
        print(f"  Run on Colab with a T4 runtime or a machine with CUDA support.")
        sys.exit(1)

    # Step 2: Build check
    print_header("BUILD CHECK")
    if not check_kernels_built():
        print(f"\n  {RED}Failed to build t4_kernels. Fix build errors above and retry.{RESET}")
        sys.exit(1)

    results = []
    total_start = time.time()

    # Step 3: Correctness tests
    if run_correctness:
        print_header("STAGE 1: CORRECTNESS TESTS")

        tests = [
            (TESTS_DIR / "test_dequant_correctness.py", "LOP3 Dequantization (U4 + S4)"),
            (TESTS_DIR / "test_fused_gemm_correctness.py", "Fused W4A16 GEMM (U4 + S4)"),
        ]

        for script, label in tests:
            if script.exists():
                print(f"\n  Running: {label}")
                passed, duration = run_test_script(script, label)
                results.append((label, passed, duration))
                print_result(label, passed, duration)
            else:
                print(f"  {YELLOW}SKIP: {script.name} not found{RESET}")
                results.append((label, False, 0.0))

    # Step 4: Benchmarks
    if run_benchmarks:
        print_header("STAGE 2: PERFORMANCE BENCHMARKS")

        bench_script = TESTS_DIR / "benchmark_kernels.py"
        if bench_script.exists():
            print(f"\n  Running: Kernel Benchmarks")
            passed, duration = run_test_script(bench_script, "Benchmarks")
            results.append(("Performance Benchmarks", passed, duration))
            print_result("Performance Benchmarks", passed, duration)
        else:
            print(f"  {YELLOW}SKIP: benchmark_kernels.py not found{RESET}")

    # Summary
    total_duration = time.time() - total_start
    print_header("FINAL SUMMARY")

    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)

    for label, passed, duration in results:
        print_result(label, passed, duration)

    print(f"\n  {BOLD}Total: {passed_count}/{total_count} passed in {total_duration:.1f}s{RESET}")

    if passed_count == total_count and total_count > 0:
        print(f"\n  {GREEN}{BOLD}>>> ALL TESTS PASSED — KERNELS VERIFIED ON REAL HARDWARE <<<{RESET}")
    elif total_count > 0:
        print(f"\n  {RED}{BOLD}>>> {total_count - passed_count} TEST(S) FAILED <<<{RESET}")

    sys.exit(0 if passed_count == total_count else 1)


if __name__ == "__main__":
    main()
