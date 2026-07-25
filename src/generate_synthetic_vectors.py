#!/usr/bin/env python3
import torch
import os

def generate_synthetic_vectors():
    out_dir = "research/scratch"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_test_vectors.pt")

    print("[Synthetic Data Generator] Generating packed INT4 weight matrices & FP16 scales...")
    torch.manual_seed(42)

    data = {}
    shapes = [1, 8, 32, 512, 4096]
    K, N = 4096, 4096

    for M in shapes:
        # Packed INT4 weights: (K // 8, N) uint32 packed
        W_packed = torch.randint(0, 0x7FFFFFFF, (K // 8, N), dtype=torch.int32)
        scale = torch.rand((N,), dtype=torch.float16) * 0.1 + 0.05
        zero = torch.randint(0, 8, (N,), dtype=torch.float16)
        A = torch.randn((M, K), dtype=torch.float16)

        data[f"M_{M}"] = {
            "A": A,
            "W_packed": W_packed,
            "scale": scale,
            "zero": zero
        }

    torch.save(data, out_path)
    print(f">> Successfully generated synthetic vectors across M={shapes}.")
    print(f">> Saved test dataset tensor package to '{out_path}'.")

if __name__ == "__main__":
    generate_synthetic_vectors()
