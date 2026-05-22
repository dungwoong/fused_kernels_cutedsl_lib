import torch
from triton.testing import do_bench
from cutedsl_kernels import Gemm5SM90
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl
import time

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
    
    a = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    ref = a @ b.t()

    # Multicast A matrix with cluster shape (1, 2)
    gemm = Gemm5SM90(
        tile_shape_mnk=(256, 128, 32),
        epi_tile_mn=(128, 128),
        cluster_shape_mnk=(1, 2, 1),
        atom_layout_mn=(2, 1),
        ab_stage=6,
        epi_stage=2,
        is_persistent=True,
        gemm_n_prologue=1,
    )

    # Multicast B matrix with cluster shape (2, 1)
    # gemm = Gemm5SM90(
    #     tile_shape_mnk=(128, 256, 32),
    #     epi_tile_mn=(128, 128),
    #     cluster_shape_mnk=(2, 1, 1),
    #     atom_layout_mn=(2, 1),
    #     ab_stage=6,
    #     epi_stage=2,
    #     is_persistent=True,
    #     gemm_n_prologue=1,
    # )

    # This works for 2048 1024 16384 1.00x
    # 2048 1024 4096 it's 0.99x
    # gemm = Gemm5SM90(
    #     tile_shape_mnk=(128, 128, 64),
    #     epi_tile_mn=(128, 32),
    #     cluster_shape_mnk=(1, 2, 1),
    #     atom_layout_mn=(2, 1),
    #     ab_stage=6,
    #     epi_stage=3,
    #     is_persistent=True,
    #     gemm_n_prologue=1,
    # )

    # Config for 4096 1536 7168 0.99x
    # gemm = Gemm5SM90(
    #     tile_shape_mnk=(192, 128, 64),
    #     epi_tile_mn=(192, 128),
    #     cluster_shape_mnk=(1, 2, 1),
    #     atom_layout_mn=(3, 1),
    #     ab_stage=4,
    #     epi_stage=1,
    #     is_persistent=True,
    #     gemm_n_prologue=0,
    # )

    # config for 3072 3072 3072 0.97x
    # gemm = Gemm5SM90(
    #     tile_shape_mnk=(192, 192, 64),
    #     epi_tile_mn=(192, 64),
    #     cluster_shape_mnk=(2, 1, 1),
    #     atom_layout_mn=(3, 1),
    #     ab_stage=3,
    #     epi_stage=3,
    #     is_persistent=True,
    #     gemm_n_prologue=0,
    # )

    # Horizontal tiles
    # gemm = Gemm5SM90(
    #     tile_shape_mnk=(64, 512, 32),
    #     epi_tile_mn=(64, 64),
    #     cluster_shape_mnk=(2, 1, 1),
    #     atom_layout_mn=(1, 2),
    #     ab_stage=5, # need lower stages cuz AB uses more SMEM
    #     epi_stage=2,
    #     is_persistent=True,
    #     gemm_n_prologue=1,
    # )
    compiled_gemm = compile_cutedsl((a, b, c), gemm, False)
    compiled_gemm(a, b, c)
    if not IS_NCU:
        print('All close:', torch.allclose(ref, c))
    if IS_DEBUG:
        # print(c)
        # print(ref)
        print((ref - c)[0, :512])
        n_incorrect = c.numel() - ((c - ref).abs() < 0.001).sum()
        print('n_incorrect :', n_incorrect)
        print('n_nonzero :', (c != 0).sum())
    
    @torch.compile
    def torch_gemm():
        return a @ b.t()
    
    def cdsl_func(a, b):
        o = torch.empty(a.shape[0], b.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a, b, o)
        return o
    
    if IS_SPEED:
        my_ms = do_bench(lambda: cdsl_func(a, b))
        time.sleep(2)
        other_ms = do_bench(torch_gemm)
        print(f'{my_ms=}, {other_ms=}')
        my_flops, other_flops = get_tflops(my_ms), get_tflops(other_ms)
        print(f'{my_flops=}, {other_flops=}')
        print(f'{other_ms/my_ms}')