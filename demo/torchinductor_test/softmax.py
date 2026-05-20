import torch
from triton.testing import do_bench
import time

t = torch.randn((4096, 4096), dtype=torch.bfloat16, device='cuda')

"""
Even for unfused, they run a single kernel called cunn_SoftMaxForwardReg
according to NCU
"""

@torch.compile
def func(a):
    return torch.softmax(a, dim=-1)

@torch.compile
def func2(a):
    e = torch.exp(a) - torch.max(a, dim=-1)[0]
    return e / torch.sum(e, dim=-1)

@torch.compile
def func3(a):
    e = torch.exp(a)
    return e / torch.sum(e, dim=-1)

# x = func(t)
# x = func(t)
# x2 = func2(t)

# print(x - x2)

ms_smx = do_bench(lambda: func(t))
# ms_smx = do_bench(lambda: torch.nn.functional.softmax(t, dim=-1))
time.sleep(2)
ms_2 = do_bench(lambda: func2(t))
time.sleep(2)
ms_3 = do_bench(lambda: func3(t))

print(f'{ms_smx/ms_2}, {ms_smx/ms_3}')

# for (4096, 4096) you have func2 is 0.65x softmax, func3 is 0.91x