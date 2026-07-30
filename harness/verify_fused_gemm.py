import torch
import time

def dequantize_u4_ref(W_packed, scale, zero_point, K, N):
    """
    Canonical PyTorch reference dequantization for W_packed (K/8 x N uint32)
    Returns FP16 matrix (K x N)
    """
    K_u32 = K // 8
    W_fp16 = torch.zeros((K, N), dtype=torch.float16, device=W_packed.device)
    
    W_cpu = W_packed.cpu().numpy()
    s_cpu = scale.cpu().numpy()
    z_cpu = zero_point.cpu().numpy()
    
    for k_idx in range(K_u32):
        for col in range(N):
            val = int(W_cpu[k_idx, col])
            s = float(s_cpu[col])
            z = float(z_cpu[col])
            for bit_i in range(8):
                u4 = (val >> (bit_i * 4)) & 0xF
                row = k_idx * 8 + bit_i
                W_fp16[row, col] = (u4 - z) * s
                
    return W_fp16

def dequantize_s4_ref(W_packed, scale, zero_point, K, N):
    """
    Canonical PyTorch reference dequantization for Signed Two's Complement INT4
    """
    K_u32 = K // 8
    W_fp16 = torch.zeros((K, N), dtype=torch.float16, device=W_packed.device)
    
    W_cpu = W_packed.cpu().numpy()
    s_cpu = scale.cpu().numpy()
    z_cpu = zero_point.cpu().numpy()
    
    for k_idx in range(K_u32):
        for col in range(N):
            val = int(W_cpu[k_idx, col])
            s = float(s_cpu[col])
            z = float(z_cpu[col])
            for bit_i in range(8):
                nib = (val >> (bit_i * 4)) & 0xF
                s4 = nib - 16 if (nib & 8) else nib
                row = k_idx * 8 + bit_i
                W_fp16[row, col] = (s4 - z) * s
                
    return W_fp16

def test_fused_gemm_correctness():
    import t4_kernels
    print("\n==========================================================================")
    print("  Tesla T4 Fused W4A16 GEMM Numerical Accuracy Verification")
    print("==========================================================================")
    
    M, K, N = 1, 1024, 1024
    print(f"Matrix Dimensions: M={M}, K={K}, N={N}")
    
    # 1. Unsigned W4A16 Test
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device='cuda')
    W_packed = torch.randint(0, 0x7FFFFFFF, (K // 8, N), dtype=torch.int32, device='cuda')
    scale = torch.rand((N,), dtype=torch.float16, device='cuda') * 0.1 + 0.05
    zero = torch.randint(0, 8, (N,), dtype=torch.float16, device='cuda')
    
    # Run Fused Kernel
    out_fused_u4 = t4_kernels.fused_w4a16_gemm_u4(A, W_packed, scale, zero)
    
    # Run Reference MatMul with FP32 accumulation matching CUDA kernel accum_f
    W_ref_u4 = dequantize_u4_ref(W_packed, scale, zero, K, N)
    out_ref_u4 = torch.matmul(A.float(), W_ref_u4.float()).half()
    
    diff_u4 = torch.max(torch.abs(out_fused_u4 - out_ref_u4)).item()
    print("Fused Output (u4):    ", out_fused_u4[0, :5])
    print("Reference Output (u4):", out_ref_u4[0, :5])
    print(f"[Unsigned U4 Fused GEMM] Max Abs Diff: {diff_u4:.5f}")
    assert diff_u4 < 2.0, f"Unsigned Fused GEMM accuracy check failed with error {diff_u4}"
    print(">> [PASSED] Unsigned Fused W4A16 GEMM Matches Reference MatMul!")
    
    # 2. Signed S4A16 Test
    out_fused_s4 = t4_kernels.fused_w4a16_gemm_s4(A, W_packed, scale, zero)
    W_ref_s4 = dequantize_s4_ref(W_packed, scale, zero, K, N)
    out_ref_s4 = torch.matmul(A.float(), W_ref_s4.float()).half()
    
    diff_s4 = torch.max(torch.abs(out_fused_s4 - out_ref_s4)).item()
    print("Fused Output (s4):    ", out_fused_s4[0, :5])
    print("Reference Output (s4):", out_ref_s4[0, :5])
    print(f"[Signed S4 Fused GEMM]   Max Abs Diff: {diff_s4:.5f}")
    assert diff_s4 < 2.0, f"Signed Fused GEMM accuracy check failed with error {diff_s4}"
    print(">> [PASSED] Signed Fused S4A16 GEMM Matches Reference MatMul!")

if __name__ == "__main__":
    test_fused_gemm_correctness()
