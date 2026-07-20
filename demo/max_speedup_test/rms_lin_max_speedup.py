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

# def do_bench(fn, iters=100):
#     for _ in range(10):
#         fn()
#     torch.cuda.synchronize()
    
#     timings = []
#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)
#     for _ in range(iters):
#         start_event.record()
#         fn()
#         end_event.record()
#         torch.cuda.synchronize()
#         curr_time = start_event.elapsed_time(end_event)
#         timings.append(curr_time)

#     mean_time = sum(timings) / iters
#     return mean_time

@torch.compile
def torch_kernel(a: torch.Tensor, b: torch.Tensor):
    a_rms = torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[1],), eps=EPS)
    return a_rms @ b.t()

@torch.compile
def torch_rmsnorm(a: torch.Tensor):
    a_rms = torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[1],), eps=EPS)
    return a_rms

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

    m, n, k = args.m, args.n, args.k
    print(f"{m=}, {n=}, {k=}")
    a = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')

    torch_kernel(a, b)

    # ms_gemm = do_bench(lambda: torch_mm(a, b))
    # time.sleep(2)
    # ms_kernel = do_bench(lambda: torch_kernel(a, b))
    # time.sleep(2)
    # ms_rms = do_bench(lambda: torch_rmsnorm(a))

    # print(f"{ms_kernel=}, {ms_gemm=}")
    # print(f"{ms_kernel / ms_gemm}")
    # print(f"{ms_rms=}, {round((ms_rms / ms_gemm) * 100, 3)}% of gemm")