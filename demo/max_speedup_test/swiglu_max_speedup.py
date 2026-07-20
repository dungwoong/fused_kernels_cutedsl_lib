import torch
from triton.testing import do_bench
from cutedsl_kernels import RMSNormLinear2SM90, RMSNormLinear1SM90
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl, STREAM
import time

"""
What's the max speedup we can get on SwiGLU?

Something around 1.08x
But also, I don't need to store back the entire results matrix

But we know I was somehow able to beat their splitk so it's not totally optimal for them...

1024 16384 1024 > 1.31x
512 16384 1024 > 1.27x
512 16384 4096 > 1.03x :(
4096 16384 4096 > 1.01x :(
4096 32768 512 > 1.54x
4096 16384 512 > 1.57x
"""

EPS = 1e-5

@torch.compile
def torch_swiglu(a, bb1):
    o1, o2 = (a @ bb1.t()).chunk(2, dim=1)
    return torch.nn.functional.silu(o1) * o2

@torch.compile
def torch_mm(a: torch.Tensor, b: torch.Tensor):
    return (a @ b.t())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int, default=4096)
    parser.add_argument("n", type=int, default=4096)
    parser.add_argument("k", type=int, default=4096)
    args = parser.parse_args()

    m, n, k = args.m, args.n, args.k
    print(f"{m=}, {n=}, {k=}")
    a = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')
    b1 = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')
    bb1 = torch.concat((b, b1), dim=0).to('cuda')

    torch_swiglu(a, bb1)
    # ms_kernel = do_bench(lambda: torch_swiglu(a, bb1))
    # time.sleep(2)
    # ms_gemm = do_bench(lambda: torch_mm(a, bb1))

    # print(f"{ms_kernel=}, {ms_gemm=}")
    # print(f"{ms_kernel / ms_gemm}")