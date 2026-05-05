import torch
from triton.testing import do_bench
from cutedsl_kernels import RMSNormLinear2SM90, RMSNormLinear1SM90
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl
import time

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
    parser.add_argument("m", type=int, default=4096)
    parser.add_argument("n", type=int, default=4096)
    parser.add_argument("k", type=int, default=4096)
    args = parser.parse_args()
    IS_NCU = args.mode == 'ncu'
    IS_DEBUG = args.mode == 'debug'
    IS_SPEED = args.mode == 'speed'

    m, n, k = args.m, args.n, args.k

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

    ckernel = RMSNormLinear2SM90(
        tile_shape_mnk=(128, 256, 32),
        epi_tile_mn=(128, 128),
        cluster_shape_mnk=(2, 1, 1),
        atom_layout_mn=(2, 1),
        ab_stage=6,
        epi_stage=2,
        is_persistent=True,
        gemm_n_prologue=0,
    )
    tensors = (a, b, c, EPS)
    compiled_cutedsl = compile_cutedsl(tensors, ckernel, False)
    compiled_cutedsl(*tensors)
    torch.cuda.synchronize()

    if not IS_NCU:
        ref_rmse = get_rmse(ref64, ref.to(htype))
        my_rmse = get_rmse(ref64, c.to(htype))
        print(f'{ref_rmse=}, {my_rmse=}')
    
    if IS_SPEED:
        def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor):
            o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
            compiled_cutedsl(a_, b_, o)
            return o
        
        my_ms = do_bench(lambda: cdsl_kernel(a, b))
        time.sleep(2)
        gemm_ms = do_bench(lambda: a @ b.t())
        time.sleep(2)
        torch_ms = do_bench(lambda: compiled_torch(a, b))
        print(f'{my_ms=}, {torch_ms=}, {gemm_ms=}')