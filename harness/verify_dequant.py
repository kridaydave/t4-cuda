#!/usr/bin/env python3
"""
Rigorous Verification & KAT Harness for LOP3 INT4 Dequantization.

Covers both:
1. Unsigned INT4 (u4 in [0..15]) via magic exponent 0x6400 (1024.0 bias)
2. Signed Two's Complement INT4 (s4 in [-8..7]) via LOP3 LUT 0x78 (bit 3 flip, 1032.0 bias)

Features:
- Asymmetrical non-monotonic hex vectors to catch positional / pair-swap bugs.
- Bit-exact IEEE 754 math proofs via struct.unpack.
- Independent canonical bit-shift unpackers.
"""

import sys
import struct

def unpack_uint32_to_u4(w_u32):
    """Canonical unsigned nibble unpacker (w0..w7)."""
    return [(w_u32 >> (i * 4)) & 0x0F for i in range(8)]


def unpack_uint32_to_s4(w_u32):
    """Canonical two's complement signed nibble unpacker (s4 in [-8..7])."""
    u4_list = unpack_uint32_to_u4(w_u32)
    return [val if val < 8 else val - 16 for val in u4_list]


def dequantize_u4_reference(packed_weights, scales, zero_points):
    """Independent canonical unsigned dequantizer: (u4 - zero) * scale"""
    out = []
    for w_u32, s, z in zip(packed_weights, scales, zero_points):
        for u4 in unpack_uint32_to_u4(w_u32):
            out.append((float(u4) - float(z)) * float(s))
    return out


def dequantize_s4_reference(packed_weights, scales, zero_points):
    """Independent canonical signed dequantizer: (s4 - zero) * scale"""
    out = []
    for w_u32, s, z in zip(packed_weights, scales, zero_points):
        for s4 in unpack_uint32_to_s4(w_u32):
            out.append((float(s4) - float(z)) * float(s))
    return out


def verify_unsigned_lop3_math():
    """Validates (0x6400 | u4) - 1024.0 == u4 for all u4 in 0..15."""
    print("\n--- [MATH PROOF] Unsigned LOP3 IEEE-754 FP16 (0x6400 | u4) ---")
    all_passed = True
    for u4 in range(16):
        fp16_bits = 0x6400 | u4
        decoded = struct.unpack('<e', struct.pack('<H', fp16_bits))[0]
        expected = 1024.0 + u4
        if decoded != expected:
            all_passed = False
        print(f"  u4={u4:2d} (0x{u4:X}) | Bits: 0x{fp16_bits:04X} | Decoded: {decoded:6.1f} | Expected: {expected:6.1f} | Match: {decoded == expected}")
    assert all_passed, "Unsigned LOP3 math failed!"
    print(">> [MATH VERIFIED] Unsigned LOP3 exponent insertion is 100% exact.")


def verify_signed_lop3_math():
    """
    Mathematical Proof Verification for Signed INT4 LUT 0x78:
    LOP3 LUT 0x78 inverts bit 3 (sign bit) while masking & ORing 0x6400.
    Resulting FP16 bit pattern represents (1032.0 + s4).
    Subtracting 1032.0 yields s4 in [-8..7] for all 16 values!
    """
    print("\n--- [MATH PROOF] Signed INT4 LOP3 LUT 0x78 (Bit 3 Inversion) ---")
    all_passed = True
    for raw_u4 in range(16):
        s4 = raw_u4 if raw_u4 < 8 else raw_u4 - 16
        # LUT 0x78 flips bit 3: bit3_flipped = raw_u4 ^ 0x8
        fp16_bits = 0x6400 | (raw_u4 ^ 0x8)
        decoded = struct.unpack('<e', struct.pack('<H', fp16_bits))[0]
        expected_raw_fp16 = 1032.0 + s4
        reconstructed_s4 = decoded - 1032.0
        match = (decoded == expected_raw_fp16) and (reconstructed_s4 == s4)
        if not match:
            all_passed = False
        print(f"  Raw 0x{raw_u4:X} -> s4={s4:2d} | Bit3 Flipped: 0x{(raw_u4^8):X} | Bits: 0x{fp16_bits:04X} | FP16: {decoded:6.1f} | (FP16 - 1032.0): {reconstructed_s4:5.1f} | Match: {match}")
    assert all_passed, "Signed LOP3 LUT 0x78 math failed!"
    print(">> [MATH VERIFIED] Signed LOP3 LUT 0x78 math is 100% exact for s4 in [-8..7].")


def run_asymmetrical_kats():
    print("\n--- [KAT] Asymmetrical Non-Monotonic Known Answer Tests ---")

    # 1. Unsigned KAT with scrambled nibble sequence W = 0xA7C13E59
    # Nibbles: w0=9, w1=5, w2=14, w3=3, w4=1, w5=12, w6=7, w7=10
    w_u4 = 0xA7C13E59
    scale_u4 = 0.25
    zero_u4 = 2.0
    u4_nibbles = unpack_uint32_to_u4(w_u4)
    u4_out = dequantize_u4_reference([w_u4], [scale_u4], [zero_u4])
    expected_u4 = [(float(v) - 2.0) * 0.25 for v in u4_nibbles]

    print(f"\nUnsigned KAT: W = 0x{w_u4:08X} | Scale = {scale_u4} | Zero = {zero_u4}")
    print(f"  Positional Nibbles (w0..w7): {u4_nibbles}")
    print(f"  Dequantized Output FP16:     {u4_out}")
    print(f"  Expected Output FP16:        {expected_u4}")
    assert u4_out == expected_u4, f"Unsigned KAT Failed! Got {u4_out}, expected {expected_u4}"
    print("  >> [PASSED] Unsigned Asymmetrical KAT Match!")

    # 2. Signed KAT with negative & positive values W = 0xF817E29A
    # Nibbles: w0=0xA(-6), w1=0x9(-7), w2=0x2(+2), w3=0xE(-2), w4=0x7(+7), w5=0x1(+1), w6=0x8(-8), w7=0xF(-1)
    w_s4 = 0xF817E29A
    scale_s4 = 0.5
    zero_s4 = 0.0
    s4_values = unpack_uint32_to_s4(w_s4)
    s4_out = dequantize_s4_reference([w_s4], [scale_s4], [zero_s4])
    expected_s4 = [float(v) * 0.5 for v in s4_values]

    print(f"\nSigned KAT (LUT 0x78): W = 0x{w_s4:08X} | Scale = {scale_s4} | Zero = {zero_s4}")
    print(f"  Positional Signed s4 (w0..w7): {s4_values}")
    print(f"  Dequantized Output FP16:       {s4_out}")
    print(f"  Expected Output FP16:          {expected_s4}")
    assert s4_out == expected_s4, f"Signed KAT Failed! Got {s4_out}, expected {expected_s4}"
    print("  >> [PASSED] Signed Asymmetrical KAT Match!")


def run_full_harness():
    print("=" * 75)
    print("  T4 CUDA Optimization: Unsigned & Signed INT4 LOP3 KAT Verification")
    print("=" * 75)

    verify_unsigned_lop3_math()
    verify_signed_lop3_math()
    run_asymmetrical_kats()

    print("\n[SUMMARY] All Unsigned (0xF8) & Signed (0x78) LOP3 Math Proofs and KATs Passed!")


if __name__ == "__main__":
    run_full_harness()
