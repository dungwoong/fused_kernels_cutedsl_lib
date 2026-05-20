import torch
from triton.testing import do_bench
from cutedsl_kernels import RMSNormLinear2SM90, RMSNormLinear1SM90
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl, STREAM
import time

"""
What's the max speedup we can get on RMSNorm+Linear?

Even for small matrix sizes e.g. 16, 4096, 4096 you get 1.15x ish
"""

EPS = 1e-5

@torch.compile
def torch_kernel(a: torch.Tensor, b: torch.Tensor):
    a_rms = torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[1],), eps=EPS)
    o1, o2 = (a_rms @ b.t()).chunk(2)
    return torch.nn.functional.silu(o1) * o2

@torch.compile
def torch_mm(a: torch.Tensor, b: torch.Tensor):
    return a @ b.t()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int, default=4096)
    parser.add_argument("n", type=int, default=4096)
    parser.add_argument("k", type=int, default=4096)
    args = parser.parse_args()

    m, n, k = args.m, 2 * args.n, args.k
    print(f"{m=}, {n=}, {k=}")
    a = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')

    ms_kernel = do_bench(lambda: torch_kernel(a, b))
    time.sleep(2)
    ms_gemm = do_bench(lambda: torch_mm(a, b))

    print(f"{ms_kernel=}, {ms_gemm=}")
    print(f"{ms_kernel / ms_gemm}")