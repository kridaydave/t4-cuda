import sys
sys.path.insert(0, '/content/repo/research/src')
import torch, t4_kernels
N = 4096*512
Wp = torch.randint(0, 0x7FFFFFFF, (N,), dtype=torch.int32, device='cuda')
scp = torch.rand((N,), dtype=torch.float16, device='cuda')
zzp = torch.zeros((N,), dtype=torch.float16, device='cuda')
for _ in range(20):
    _ = t4_kernels.dequantize_s4(Wp, scp, zzp)
torch.cuda.synchronize()
