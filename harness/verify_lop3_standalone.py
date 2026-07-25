import torch
import struct
import numpy as np

# Ensure PyTorch CUDA is available
assert torch.cuda.is_available(), "CUDA device required for verification"

import t4_kernels

def python_unpack_u4(w_u32, scale, zero_point):
    """
    Python reference for Unsigned INT4 Dequantization:
    W = 8 nibbles packed in u32: [w0, w1, w2, w3, w4, w5, w6, w7]
    Dequantized = (u4 - zero_point) * scale
    """
    nibbles = [(w_u32 >> (i * 4)) & 0xF for i in range(8)]
    # Dequantized values
    return [(n - zero_point) * scale for n in nibbles]

def python_unpack_s4(w_u32, scale, zero_point):
    """
    Python reference for Signed Two's Complement INT4 Dequantization:
    s4 in [-8, 7]
    Dequantized = (s4 - zero_point) * scale
    """
    nibbles = [(w_u32 >> (i * 4)) & 0xF for i in range(8)]
    s4_vals = []
    for n in nibbles:
        # 4-bit two's complement decoding: if bit 3 set, subtract 16
        s4 = n - 16 if (n & 8) else n
        s4_vals.append(s4)
    return [(s - zero_point) * scale for s in s4_vals]

def test_lop3_standalone_u4():
    print("\n--- [STEP 1A] Standalone LOP3 0xEA Unsigned INT4 Dequantization Unit Test ---")
    
    # Test case: 4 packed uint32 words (32 weights total)
    packed_weights = torch.tensor([0xA7C13E59, 0x12345678, 0xFEDCBA98, 0x0F0F0F0F], dtype=torch.int32, device='cuda')
    scales = torch.tensor([0.25, 0.5, 0.1, 1.0], dtype=torch.float16, device='cuda')
    zeros = torch.tensor([2.0, 0.0, 1.0, 0.0], dtype=torch.float16, device='cuda')
    
    # Run CUDA kernel
    gpu_output = t4_kernels.dequantize_u4(packed_weights, scales, zeros)
    
    # Compute Python reference
    cpu_packed = packed_weights.cpu().numpy()
    cpu_scales = scales.cpu().numpy()
    cpu_zeros = zeros.cpu().numpy()
    
    ref_vals = []
    for i, w in enumerate(cpu_packed):
        ref_vals.extend(python_unpack_u4(w, cpu_scales[i], cpu_zeros[i]))
        
    ref_tensor = torch.tensor(ref_vals, dtype=torch.float16, device='cuda')
    
    # Output comparison
    diff = torch.max(torch.abs(gpu_output - ref_tensor)).item()
    print(f"GPU Output (u4):  {gpu_output[:8].tolist()}")
    print(f"Ref Output (u4):  {ref_tensor[:8].tolist()}")
    print(f"Max Absolute Error: {diff:.6f}")
    assert diff < 1e-3, f"Unsigned LOP3 unit test failed with error {diff}"
    print(">> [PASSED] Unsigned LOP3 0xEA Unpack Unit Test Exact Match!")

def test_lop3_standalone_s4():
    print("\n--- [STEP 1B] Standalone LOP3 0x6A Signed INT4 Dequantization Unit Test ---")
    
    packed_weights = torch.tensor([0xF817E29A, 0x87654321, 0x98BADCFE, 0xF0F0F0F0], dtype=torch.int32, device='cuda')
    scales = torch.tensor([0.5, 0.25, 0.2, 0.5], dtype=torch.float16, device='cuda')
    zeros = torch.tensor([0.0, 1.0, -1.0, 0.0], dtype=torch.float16, device='cuda')
    
    # Run CUDA kernel
    gpu_output = t4_kernels.dequantize_s4(packed_weights, scales, zeros)
    
    # Compute Python reference
    cpu_packed = packed_weights.cpu().numpy()
    cpu_scales = scales.cpu().numpy()
    cpu_zeros = zeros.cpu().numpy()
    
    ref_vals = []
    for i, w in enumerate(cpu_packed):
        ref_vals.extend(python_unpack_s4(w, cpu_scales[i], cpu_zeros[i]))
        
    ref_tensor = torch.tensor(ref_vals, dtype=torch.float16, device='cuda')
    
    # Output comparison
    diff = torch.max(torch.abs(gpu_output - ref_tensor)).item()
    print(f"GPU Output (s4):  {gpu_output[:8].tolist()}")
    print(f"Ref Output (s4):  {ref_tensor[:8].tolist()}")
    print(f"Max Absolute Error: {diff:.6f}")
    assert diff < 1e-3, f"Signed LOP3 unit test failed with error {diff}"
    print(">> [PASSED] Signed LOP3 0x6A Unpack Unit Test Exact Match!")

if __name__ == "__main__":
    print("==========================================================================")
    print("  Tesla T4 LOP3 Standalone Dequantization Isolation Unit Test Harness")
    print("==========================================================================")
    test_lop3_standalone_u4()
    test_lop3_standalone_s4()
    print("\n==========================================================================")
    print("  [SUCCESS] All Isolated LOP3 Unpack Tests Passed!")
    print("==========================================================================")
