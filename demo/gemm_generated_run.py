import torch
import time
from triton.testing import do_bench
from cutedsl_kernels.experimental.gemm_generated import Kernel
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl


if __name__ == '__main__':
    print('Starting...')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["debug", "speed", "ncu"])
    args = parser.parse_args()
    IS_NCU = args.mode == 'ncu'
    IS_DEBUG = args.mode == 'debug'
    IS_SPEED = args.mode == 'speed'

    m, n, k = 4096, 4096, 4096
    
    a = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    ref = a @ b.t()

    gemm = Kernel()
    compiled_gemm = compile_cutedsl((a, b, c), gemm, False)
    compiled_gemm(a, b, c)
    print(c - ref)
    print('allclose:', torch.allclose(ref, c))

    def gemm_fn(a, b):
        c_ = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a, b, c_)
    my_ms = do_bench(lambda: gemm_fn(a, b))
    time.sleep(2)
    torch_ms = do_bench(lambda: a @ b.t())
    print(f'{my_ms=}, {torch_ms=}, {torch_ms / my_ms}')