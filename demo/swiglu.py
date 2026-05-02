import torch
from triton.testing import do_bench
from cutedsl_kernels import Swiglu2SM90
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl

def get_rmse(ref: torch.Tensor, o: torch.Tensor):
    assert o.dtype == ref.dtype
    mse = torch.nn.functional.mse_loss(o, ref, reduction='mean')
    rmse = mse.sqrt().item()
    return rmse

# TODO ncu setup is messed up now
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
    flops = 2 * m * n * k

    def get_tflops(time_ms):
        return (flops / (time_ms / 1e3)) / 1e12
    
    a64 = torch.randn((m, k), dtype=torch.float64)
    b64 = torch.randn((n, k), dtype=torch.float64)
    b164 = torch.randn((n, k), dtype=torch.float64)
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    bb164 = torch.cat((b64, b164), dim=0).to('cuda')

    a = a64.to(torch.bfloat16).to('cuda')
    b = b64.to(torch.bfloat16).to('cuda')
    b1 = b164.to(torch.bfloat16).to('cuda')
    bb1 = bb164.to(torch.bfloat16).to('cuda')
    
    a64 = a64.to('cuda')
    b64 = b64.to('cuda')
    b164 = b164.to('cuda')

    def torch_swiglu(a, bb1):
        o1, o2 = (a @ bb1.t()).chunk(2, dim=1)
        return torch.nn.functional.silu(o1) * o2
    
    ref_64 = torch_swiglu(a64, bb164)
    ref = torch_swiglu(a, bb1)

    gemm = Swiglu2SM90(
        tile_shape_mnk=(128, 128, 32),
        epi_tile_mn=(128, 32),
        cluster_shape_mnk=(2, 1, 1),
        atom_layout_mn=(2, 1),
        ab_stage=6,
        epi_stage=2,
        reuse_ab=False,
        is_persistent=True,
        gemm_n_prologue=1,
    )
    compiled_gemm = compile_cutedsl((a, b, b1, c), gemm, False)
    compiled_gemm(a, b, b1, c)
    if not IS_NCU:
        rmse_ref = get_rmse(ref.to(ref_64.dtype), ref_64)
        rmse_mine = get_rmse(c.to(ref_64.dtype), ref_64)
        print(f'{rmse_ref=}, {rmse_mine=}')
    
    torch_func = torch.compile(torch_swiglu)
    
    def cdsl_func(a, b, b1):
        o = torch.empty(a.shape[0], b.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a, b, b1, o)
        return o
    
    if IS_SPEED:
        my_ms = do_bench(lambda: cdsl_func(a, b, b1))
        other_ms = do_bench(lambda: torch_func(a, bb1))
        print(f'{my_ms=}, {other_ms=}')
        my_flops, other_flops = get_tflops(my_ms), get_tflops(other_ms)
        print(f'{my_flops=}, {other_flops=}')
        print(f'{other_ms / my_ms}')