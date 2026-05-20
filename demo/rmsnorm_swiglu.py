import time
import torch
from triton.testing import do_bench
from cutedsl_kernels import RMSNormSwiglu1
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl

EPS = 1e-5

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

    def torch_fn(a, bb1):
        a = torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[1],), eps=EPS)
        o1, o2 = (a @ bb1.t()).chunk(2, dim=1)
        return torch.nn.functional.silu(o1) * o2
    
    def torch_fn_slow(a, b, b1):
        a = torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[1],), eps=EPS)
        o1 = a @ b.t()
        o2 = a @ b1.t()
        return torch.nn.functional.silu(o1) * o2
    
    ref_64 = torch_fn(a64, bb164)
    ref = torch_fn(a, bb1)

    gemm = RMSNormSwiglu1(
        tile_shape_mnk=(128, 128, 64),
        epi_tile_mn=(128, 32),
        cluster_shape_mnk=(2, 1, 1),
        atom_layout_mn=(2, 1),
        ab_stage=3,
        epi_stage=2,
        is_persistent=True,
        gemm_n_prologue=0,
    )

    # this is decent on 64 16384 4096
    # gemm = RMSNormSwiglu1(
    #     tile_shape_mnk=(64, 128, 64),
    #     epi_tile_mn=(64, 32),
    #     cluster_shape_mnk=(2, 1, 1),
    #     atom_layout_mn=(1, 1),
    #     ab_stage=5,
    #     epi_stage=3,
    #     is_persistent=True,
    #     gemm_n_prologue=0,
    # )
    compiled_gemm = compile_cutedsl((a, b, b1, c, EPS), gemm, False)
    compiled_gemm(a, b, b1, c, EPS)
    if not IS_NCU:
        rmse_ref = get_rmse(ref.to(ref_64.dtype), ref_64)
        rmse_mine = get_rmse(c.to(ref_64.dtype), ref_64)
        print(f'{rmse_ref=}, {rmse_mine=}')
    
    torch_func = torch.compile(torch_fn)
    # torch_func = torch_fn
    torch_func_slow = torch.compile(torch_fn_slow)
    
    def cdsl_func(a, b, b1):
        o = torch.empty(a.shape[0], b.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a, b, b1, o, EPS)
        return o
    
    if IS_SPEED:
        my_ms = do_bench(lambda: cdsl_func(a, b, b1))
        time.sleep(2)
        other_ms = do_bench(lambda: torch_func(a, bb1))
        time.sleep(2)
        other_ms_slow = do_bench(lambda: torch_func_slow(a, b, b1))
        time.sleep(2)
        ms_gemm = do_bench(lambda: a @ bb1.t())
        print(f'{my_ms=}, {other_ms=}, {other_ms_slow=}')
        print(f'fast ver speedup(RS wgmma) = {other_ms / my_ms}')
        # print(f'fast ver speedup = {other_ms / my_ms}')
        # print(f'slow ver speedup = {other_ms_slow / my_ms}')
        print(f'speedup rs over gemm = {ms_gemm / my_ms}')
        print(f'max potential fast ver speedup = {other_ms / ms_gemm}')