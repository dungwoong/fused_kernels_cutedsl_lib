import torch
import time
from triton.testing import do_bench
from cutedsl_kernels.experimental.swiglu import Kernel
from cutedsl_kernels.experimental.swiglu_high_level_generated import Kernel as HLKernel
from cutedsl_kernels.experimental.swiglu_hel import Kernel as HelKernel
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl

EPS = 1e-5

def get_rmse(ref: torch.Tensor, o: torch.Tensor):
    assert o.dtype == ref.dtype
    mse = torch.nn.functional.mse_loss(o, ref, reduction='mean')
    rmse = mse.sqrt().item()
    return rmse

def torch_kernel(a, b, b1):
    o1 = a @ b.t()
    o2 = a @ b1.t()
    return torch.nn.functional.silu(o1) * o2

if __name__ == '__main__':
    print('Starting...')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int, default=4096)
    parser.add_argument("n", type=int, default=4096)
    parser.add_argument("k", type=int, default=4096)
    parser.add_argument("--mode", type=int, choices=[0, 1, 2], default=0)
    args = parser.parse_args()

    m, n, k = args.m, args.n, args.k
    
    htype = torch.float64
    dtype = torch.bfloat16
    a64 = torch.randn((m, k), dtype=htype).to('cuda')
    b64 = torch.randn((n, k), dtype=htype).to('cuda')
    b164 = torch.randn((n, k), dtype=htype).to('cuda')

    a = a64.to(dtype)
    b = b64.to(dtype)
    b1 = b164.to(dtype)
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    ref64 = torch_kernel(a64, b64, b164)

    compiled_torch = torch.compile(torch_kernel)
    ref = compiled_torch(a, b, b1)

    print('Reference finished')
    gemm_classes = {
        0: HLKernel,
        1: Kernel,
        2: HelKernel,
    }
    GemmCls = gemm_classes[args.mode]
    gemm = GemmCls()
    # gemm = Kernel()
    print('Compiling kernel')
    compiled_gemm = compile_cutedsl((a, b, b1, c), gemm, False)
    print('Running gemm')
    compiled_gemm(a, b, b1, c)
    
    torch_rmse = get_rmse(ref64, ref.to(htype))
    my_rmse = get_rmse(ref64, c.to(htype))
    print(f'{torch_rmse=} {my_rmse=}')

    def cdsl_fn(a, b, b1):
        c_ = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a, b, b1, c_)

    my_ms = do_bench(lambda: cdsl_fn(a, b, b1))
    time.sleep(2)
    torch_ms = do_bench(lambda: compiled_torch(a, b, b1))
    print(f'{my_ms=}, {torch_ms=}, {torch_ms / my_ms}')