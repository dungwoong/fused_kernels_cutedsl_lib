import torch
from triton.testing import do_bench
import time

t = torch.randn((4096, 4096), dtype=torch.bfloat16, device='cuda')

"""
Even for unfused, they run a single kernel called cunn_SoftMaxForwardReg
according to NCU
"""
EPS = 1e-5

@torch.compile
def func(a):
    return torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[-1],), eps=EPS)

@torch.compile
def func2(a):
    denom = torch.rsqrt(torch.mean(torch.square(a), dim=-1, keepdim=True) + EPS)
    return a * denom

# x = func(t)
# x2 = func2(t)

# print(x)
# print(x2)
# print(x - x2)

ms_rms = do_bench(lambda: func(t))
time.sleep(2)
ms_2 = do_bench(lambda: func2(t))
print(f'({ms_rms=}, {ms_2=}), {ms_rms/ms_2}')

# 0.999x