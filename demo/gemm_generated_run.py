import torch
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
    compiled_gemm = compile_cutedsl((a, b), gemm, False)
    compiled_gemm(a, b)