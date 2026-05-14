import torch
from triton.testing import do_bench
from cutedsl_kernels import RMSNormLinear2SM90, RMSNormLinear1SM90
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl, STREAM
import time

"""
What's the max speedup we can get on LoRA?

if we can fit everything into a single GEMM kernel, we could get large-ish speedups(I'm comparing to only XW gemm)

you get moderate speedups if you don't tho

I test against xW gemm only or xW + xA timing

mnk gemm only / gemm + xA
16 16384 4096 1.30/1.12

16 4096  4096 1.65/1.19

16 2048  2048 2.17/1.36

2048 2048 2048 1.74/1.37 I got 1.34x so that's close

4096 4096 4096 1.30/1.21. With my formulation I got 1.19x which is close to 1.21. That makes sense.

Ok I think there could be some potential here but these max speedups are unrealistic so...
"""

EPS = 1e-5

@torch.compile
def torch_lora(x, w, a, b):
    return x @ w.t() + (x @ a) @ b
    

def torch_mm2(x, w, a, b):
    return x @ w.t(), x @ a

def torch_mm(x, w, a, b):
    return x @ w.t()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int, default=4096)
    parser.add_argument("n", type=int, default=4096)
    parser.add_argument("k", type=int, default=4096)
    lora_dim=16
    args = parser.parse_args()

    m, n, k = args.m, args.n, args.k
    print(f"{m=}, {n=}, {k=}")
    x = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    w = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')
    a = torch.randn((k, lora_dim), dtype=torch.bfloat16).to('cuda')
    b = torch.randn((lora_dim, n), dtype=torch.bfloat16).to('cuda')

    ms_kernel = do_bench(lambda: torch_lora(x, w, a, b))
    time.sleep(2)
    ms_gemm = do_bench(lambda: torch_mm(x, w, a, b))
    time.sleep(2)
    ms_gemm2 = do_bench(lambda: torch_mm2(x, w, a, b))

    print(f"{ms_kernel=}, {ms_gemm=}, {ms_gemm2}")
    print(f"{ms_kernel / ms_gemm}, {ms_kernel / ms_gemm2}")