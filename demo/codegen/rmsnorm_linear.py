import torch
import time
from triton.testing import do_bench
from cutedsl_kernels.experimental.gemm_generated import Kernel
from cutedsl_kernels.experimental.gemm_persistent_4096 import Kernel as GemmPersistent
from cutedsl_kernels.experimental.rmsnorm_linear_4096 import Kernel as RMSLinKernel
from cutedsl_kernels.experimental.rmsnorm_linear_high_level_generated import Kernel as HLKernel
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl

EPS = 1e-5

def get_rmse(ref: torch.Tensor, o: torch.Tensor):
    assert o.dtype == ref.dtype
    mse = torch.nn.functional.mse_loss(o, ref, reduction='mean')
    rmse = mse.sqrt().item()
    return rmse

def torch_kernel(a: torch.Tensor, b: torch.Tensor):
    a_rms = torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[1],), eps=EPS)
    return a_rms @ b.t()


if __name__ == '__main__':
    print('Starting...')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["debug", "speed", "ncu"])
    args = parser.parse_args()
    IS_NCU = args.mode == 'ncu'
    IS_DEBUG = args.mode == 'debug'
    IS_SPEED = args.mode == 'speed'

    m, n, k = 4096, 1024, 4096
    
    htype = torch.float64
    dtype = torch.bfloat16
    a64 = torch.randn((m, k), dtype=htype).to('cuda')
    b64 = torch.randn((n, k), dtype=htype).to('cuda')

    a = a64.to(dtype)
    b = b64.to(dtype)
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    ref64 = torch_kernel(a64, b64)

    compiled_torch = torch.compile(torch_kernel)
    ref = compiled_torch(a, b)

    print('Reference finished')
    # gemm = RMSLinKernel()
    gemm = HLKernel()
    print('Compiling kernel')
    compiled_gemm = compile_cutedsl((a, b, c), gemm, False)
    print('Running gemm')
    compiled_gemm(a, b, c)
    
    torch_rmse = get_rmse(ref64, ref.to(htype))
    my_rmse = get_rmse(ref64, c.to(htype))
    print(f'{torch_rmse=} {my_rmse=}')

    def cdsl_fn(a, b):
        c_ = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a, b, c_)

    my_ms = do_bench(lambda: cdsl_fn(a, b))
    time.sleep(2)
    torch_ms = do_bench(lambda: compiled_torch(a, b))
    print(f'{my_ms=}, {torch_ms=}, {torch_ms / my_ms}')