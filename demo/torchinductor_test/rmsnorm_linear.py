import torch
from triton.testing import do_bench
import time

EPS=1e-5

@torch.compile
def torch_kernel(a: torch.Tensor, b: torch.Tensor):
    a_rms = torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[1],), eps=EPS)
    return a_rms @ b.t()

@torch.compile
def fn2(a, b):
    denom = torch.rsqrt(torch.mean(torch.square(a), dim=-1, keepdim=True) + EPS)
    return (a @ b.t()) * denom

a = torch.randn((4096, 4096), dtype=torch.bfloat16, device='cuda')
b = torch.randn((4096, 4096), dtype=torch.bfloat16, device='cuda')
# o1 = torch_kernel(a, b)
# o2 = fn2(a, b)
# print((o1 - o2).abs().max())

my_ms = do_bench(lambda: torch_kernel(a, b))
time.sleep(2)
other_ms = do_bench(lambda: fn2(a, b))

print(f'{my_ms / other_ms}') # 0.98x