import sys
import torch
import numpy as np

try:
    import t4_kernels
except ImportError:
    print("ERROR: t4_kernels not found. Please run build_and_check.sh first.")
    sys.exit(1)

# Ensure PyTorch uses the GPU
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. This test requires a GPU.")
    sys.exit(1)

device = torch.device('cuda')

def cpu_dequantize(packed_tensor, scales_tensor, zps_tensor, is_signed=False):
    """
    Pure CPU/NumPy reference implementation for 4-bit dequantization.
    """
    # Move to CPU and use uint32 for accurate bitwise operations
    packed = packed_tensor.detach().cpu().numpy().astype(np.uint32)
    scales = scales_tensor.detach().cpu().numpy().astype(np.float32)
    zps = zps_tensor.detach().cpu().numpy().astype(np.float32)
    
    if scales.size == 1:
        scales = np.broadcast_to(scales, packed.shape)
    if zps.size == 1:
        zps = np.broadcast_to(zps, packed.shape)
        
    # Unpack 8 nibbles (lowest 4 bits first)
    unpacked = np.zeros((*packed.shape, 8), dtype=np.float32)
    for i in range(8):
        nibble = (packed >> (i * 4)) & 0xF
        if is_signed:
            # Sign extension for 4-bit (two's complement)
            nibble = np.where(nibble >= 8, nibble - 16, nibble)
        
        # Apply formula: value = (nibble_value - zero_point) * scale
        unpacked[..., i] = (nibble - zps) * scales

    return torch.tensor(unpacked, dtype=torch.float16, device=packed_tensor.device)

def run_test_case(name, packed_vals, scales, zps, is_signed):
    """
    Runs a single test case, compares GPU vs CPU, and prints the result.
    """
    try:
        if is_signed:
            gpu_out = t4_kernels.dequantize_s4(packed_vals, scales, zps)
        else:
            gpu_out = t4_kernels.dequantize_u4(packed_vals, scales, zps)
            
        # Ensure outputs are comparable shapes (N, 8)
        gpu_out = gpu_out.view(-1, 8)
        cpu_out = cpu_dequantize(packed_vals, scales, zps, is_signed).view(-1, 8)
        
        # Calculate errors
        abs_diff = torch.abs(gpu_out - cpu_out)
        max_err = abs_diff.max().item()
        mean_err = abs_diff.float().mean().item()
        
        # Compare with tolerance suitable for FP16
        is_pass = torch.allclose(gpu_out, cpu_out, atol=1e-2, rtol=1e-3)
        
        status = "\033[92mPASS\033[0m" if is_pass else "\033[91mFAIL\033[0m"
        print(f"{name:<35} | {status} | Max Err: {max_err:.4e} | Mean Err: {mean_err:.4e}")
        return is_pass
        
    except Exception as e:
        print(f"{name:<35} | \033[91mERROR\033[0m | {str(e)}")
        return False

def main():
    print("=" * 80)
    print("CUDA Dequantize (u4 & s4) Correctness Tests")
    print("=" * 80)
    print(f"{'Test Name':<35} | Status | Max Error    | Mean Error")
    print("-" * 80)
    
    total_tests = 0
    passed_tests = 0

    def test(name, packed_vals, scales, zps, is_signed=False):
        nonlocal total_tests, passed_tests
        total_tests += 1
        if run_test_case(name, packed_vals, scales, zps, is_signed):
            passed_tests += 1

    # Single value helper
    def make_tensors(packed, scale, zp, batch=1):
        p_tensor = torch.full((batch,), packed, dtype=torch.int32, device=device)
        s_tensor = torch.full((batch,), scale, dtype=torch.float16, device=device)
        z_tensor = torch.full((batch,), zp, dtype=torch.float16, device=device)
        return p_tensor, s_tensor, z_tensor

    # 1. KAT Test (H4)
    # 0xA7C13E59, scale=0.25, zp=2.0
    kat_val = np.uint32(0xA7C13E59).view(np.int32)
    p, s, z = make_tensors(kat_val, 0.25, 2.0)
    test("KAT Test (H4) - u4", p, s, z, is_signed=False)
    test("KAT Test (H4) - s4", p, s, z, is_signed=True)

    # 2. KAT Test (All Zeros)
    p, s, z = make_tensors(0x00000000, 1.0, 0.0)
    test("KAT Test (All Zeros) - u4", p, s, z, is_signed=False)
    test("KAT Test (All Zeros) - s4", p, s, z, is_signed=True)

    # 3. KAT Test (All Ones)
    # 0xFFFFFFFF
    ones_val = np.uint32(0xFFFFFFFF).view(np.int32)
    p, s, z = make_tensors(ones_val, 1.0, 0.0)
    test("KAT Test (All Ones) - u4", p, s, z, is_signed=False)
    test("KAT Test (All Ones) - s4", p, s, z, is_signed=True)

    # 4. KAT Test (Identity)
    # 0x76543210
    id_val = np.uint32(0x76543210).view(np.int32)
    p, s, z = make_tensors(id_val, 1.0, 0.0)
    test("KAT Test (Identity) - u4", p, s, z, is_signed=False)
    test("KAT Test (Identity) - s4", p, s, z, is_signed=True)

    # 5. Exhaustive Nibble Sweep (0-15)
    for n in range(16):
        # Pack the nibble 8 times into a uint32
        val = sum(n << (i * 4) for i in range(8))
        val_i32 = np.uint32(val).view(np.int32)
        p, s, z = make_tensors(val_i32, 1.0, 0.0)
        # We'll group them into one test call for cleaner output, but doing it in bulk is better
    
    sweep_vals = []
    for n in range(16):
        sweep_vals.append(sum(n << (i * 4) for i in range(8)))
    sweep_tensor = torch.tensor(np.array(sweep_vals, dtype=np.uint32).view(np.int32), device=device)
    s_sweep = torch.ones((16,), dtype=torch.float16, device=device)
    z_sweep = torch.zeros((16,), dtype=torch.float16, device=device)
    test("Exhaustive Nibble Sweep - u4", sweep_tensor, s_sweep, z_sweep, is_signed=False)
    test("Exhaustive Nibble Sweep - s4", sweep_tensor, s_sweep, z_sweep, is_signed=True)

    # 6. Random Fuzz
    fuzz_size = 10000
    fuzz_p = torch.randint(-2147483648, 2147483647, (fuzz_size,), dtype=torch.int32, device=device)
    fuzz_s = torch.randn(fuzz_size, dtype=torch.float16, device=device) * 5.0
    fuzz_z = torch.randn(fuzz_size, dtype=torch.float16, device=device) * 5.0
    test("Random Fuzz (10K) - u4", fuzz_p, fuzz_s, fuzz_z, is_signed=False)
    test("Random Fuzz (10K) - s4", fuzz_p, fuzz_s, fuzz_z, is_signed=True)

    # 7. Batch Size Stress
    for b in [1, 256, 4096, 1000000]:
        bp, bs, bz = make_tensors(id_val, 0.5, 1.0, batch=b)
        test(f"Batch Size Stress ({b}) - u4", bp, bs, bz, is_signed=False)
        test(f"Batch Size Stress ({b}) - s4", bp, bs, bz, is_signed=True)

    # 8. Scale Edge Cases
    edge_p, _, _ = make_tensors(id_val, 1.0, 0.0, batch=1)
    
    # Scale = 0
    zero_s = torch.tensor([0.0], dtype=torch.float16, device=device)
    zero_z = torch.tensor([0.0], dtype=torch.float16, device=device)
    test("Scale Edge (scale=0) - u4", edge_p, zero_s, zero_z, is_signed=False)
    
    # Scale = very large
    large_s = torch.tensor([65000.0], dtype=torch.float16, device=device) # close to fp16 max
    test("Scale Edge (scale=65k) - u4", edge_p, large_s, zero_z, is_signed=False)
    
    # Scale = very small
    small_s = torch.tensor([1e-5], dtype=torch.float16, device=device)
    test("Scale Edge (scale=1e-5) - u4", edge_p, small_s, zero_z, is_signed=False)

    print("=" * 80)
    print(f"SUMMARY: {passed_tests}/{total_tests} Tests Passed")
    print("=" * 80)
    
    if passed_tests < total_tests:
        sys.exit(1)

if __name__ == "__main__":
    main()
