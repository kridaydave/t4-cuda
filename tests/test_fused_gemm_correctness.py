import torch
import sys

try:
    import t4_kernels
    HAS_T4_KERNELS = True
except ImportError:
    HAS_T4_KERNELS = False
    print("WARNING: t4_kernels not found. Tests will skip CUDA executions.")

def unpack_u4(W_packed, K, N):
    unpacked = torch.zeros((K, N), dtype=torch.uint8)
    for i in range(8):
        unpacked[i::8, :] = (W_packed >> (i * 4)) & 0xF
    return unpacked

def unpack_s4(W_packed, K, N):
    unpacked = unpack_u4(W_packed, K, N).to(torch.int8)
    unpacked[unpacked >= 8] -= 16
    return unpacked

def dequantize_ref(W_packed, scales, zero_points, signed=False):
    K_8, N = W_packed.shape
    K = K_8 * 8
    if signed:
        unpacked = unpack_s4(W_packed, K, N).float()
    else:
        unpacked = unpack_u4(W_packed, K, N).float()
    
    W_dequant = (unpacked - zero_points.float()) * scales.float()
    return W_dequant.half()

def cpu_reference_gemm(A, W_packed, scales, zero_points, signed=False):
    W_dequant = dequantize_ref(W_packed, scales, zero_points, signed=signed)
    C_ref = torch.matmul(A.float(), W_dequant.float())
    return C_ref.half()

def run_test_case(M, K, N, name, signed=False, tol=0.5):
    print(f"Running Test: {name} (M={M}, K={K}, N={N}, Signed={signed})")
    
    # Generate random inputs
    A = torch.randn(M, K, dtype=torch.float16, device='cuda' if HAS_T4_KERNELS else 'cpu')
    W_packed = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32, device='cuda' if HAS_T4_KERNELS else 'cpu')
    scales = torch.randn(1, N, dtype=torch.float16, device='cuda' if HAS_T4_KERNELS else 'cpu') * 0.1
    zero_points = torch.randint(0, 16, (1, N), dtype=torch.float16, device='cuda' if HAS_T4_KERNELS else 'cpu')

    C_ref = cpu_reference_gemm(A.cpu(), W_packed.cpu(), scales.cpu(), zero_points.cpu(), signed=signed)

    if not HAS_T4_KERNELS:
        print(f"  [SKIPPED] Missing t4_kernels")
        return True

    if signed:
        C_out = t4_kernels.fused_w4a16_gemm_s4(A, W_packed, scales, zero_points)
    else:
        C_out = t4_kernels.fused_w4a16_gemm_u4(A, W_packed, scales, zero_points)

    max_err = torch.max(torch.abs(C_out.cpu() - C_ref)).item()
    mean_err = torch.mean(torch.abs(C_out.cpu() - C_ref)).item()
    
    if max_err > tol:
        print(f"  [FAIL] Max Err: {max_err:.4f}, Mean Err: {mean_err:.4f}")
        return False
    else:
        print(f"  [PASS] Max Err: {max_err:.4f}, Mean Err: {mean_err:.4f}")
        return True

def test_identity():
    print("Running Test: Identity Matrix")
    K = 8
    N = 8
    A = torch.eye(K, dtype=torch.float16)
    W_packed = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32)
    scales = torch.ones(1, N, dtype=torch.float16)
    zero_points = torch.zeros(1, N, dtype=torch.float16)
    
    C_ref = cpu_reference_gemm(A, W_packed, scales, zero_points, signed=False)
    W_dequant = dequantize_ref(W_packed, scales, zero_points, signed=False)
    
    max_err = torch.max(torch.abs(C_ref - W_dequant)).item()
    if max_err < 1e-3:
        print("  [PASS] Identity matched weights exactly")
    else:
        print(f"  [FAIL] Identity mismatch: max err {max_err}")

def run_all_tests():
    test_identity()
    run_test_case(1, 8, 4, "Known Small Matrix", signed=False)
    run_test_case(1, 3584, 3584, "Eli Model Sizes (Decode)", signed=False)
    for b in [1, 2, 4, 8]:
        run_test_case(b, 2048, 4096, f"Batch Decode M={b}", signed=True)
    
    print("\nRunning Random Fuzz...")
    max_errors = []
    mean_errors = []
    for _ in range(100):
        # Run silent
        A = torch.randn(1, 512, dtype=torch.float16)
        W_packed = torch.randint(0, 2**31 - 1, (512 // 8, 512), dtype=torch.int32)
        scales = torch.randn(1, 512, dtype=torch.float16) * 0.1
        zero_points = torch.randint(0, 16, (1, 512), dtype=torch.float16)
        C_ref = cpu_reference_gemm(A, W_packed, scales, zero_points, signed=False)
        
        if HAS_T4_KERNELS:
            C_out = t4_kernels.fused_w4a16_gemm_u4(A.cuda(), W_packed.cuda(), scales.cuda(), zero_points.cuda())
            max_err = torch.max(torch.abs(C_out.cpu() - C_ref)).item()
            mean_err = torch.mean(torch.abs(C_out.cpu() - C_ref)).item()
            max_errors.append(max_err)
            mean_errors.append(mean_err)
    if HAS_T4_KERNELS:
        print(f"  Fuzz 100 iterations -> Max Err: {max(max_errors):.4f}, Avg Mean Err: {sum(mean_errors)/100:.4f}")
    
    run_test_case(1, 8, 1, "Edge Case: Minimum K, Single N")
    
if __name__ == '__main__':
    run_all_tests()
