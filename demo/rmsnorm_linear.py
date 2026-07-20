import torch
from triton.testing import do_bench
from cutedsl_kernels import RMSNormLinear2SM90, RMSNormLinear1SM90
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl, STREAM
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

@torch.compile
def gemm_kernel(a: torch.Tensor, b: torch.Tensor):
    return a @ b.t()

# ncu --set full -o rmslin_4096_1536_7168 -f --launch-skip 8 python3 demo/rmsnorm_linear.py ncu 4096 1536 7168
if __name__ == '__main__':
    print('Starting...')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["debug", "speed", "ncu"])
    parser.add_argument("m", type=int, default=4096)
    parser.add_argument("n", type=int, default=4096)
    parser.add_argument("k", type=int, default=4096)
    parser.add_argument("--do-old", action='store_true')
    args = parser.parse_args()
    IS_NCU = args.mode == 'ncu'
    IS_DEBUG = args.mode == 'debug'
    IS_SPEED = args.mode == 'speed'
    DO_OLD = args.do_old

    m, n, k = args.m, args.n, args.k

    htype = torch.float64
    dtype = torch.bfloat16
    a64 = torch.randn((m, k), dtype=htype).to('cuda')
    b64 = torch.randn((n, k), dtype=htype).to('cuda')

    a = a64.to(dtype)
    b = b64.to(dtype)
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    c_old = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    
    ref64 = torch_kernel(a64, b64)
    compiled_torch = torch.compile(torch_kernel)
    ref = compiled_torch(a, b)

    ckernel = RMSNormLinear2SM90(
        tile_shape_mnk=(128, 256, 64),
        epi_tile_mn=(128, 64),
        cluster_shape_mnk=(1, 2, 1),
        atom_layout_mn=(2, 1),
        ab_stage=4,
        epi_stage=2,
        is_persistent=True,
        gemm_n_prologue=0,
        pingpong=False,
    )

    # 4096 1536 7168
    if (m, n, k) == (4096, 1536, 7168):
        print('Using specialized config')
        ckernel = RMSNormLinear2SM90(
            tile_shape_mnk=(192, 128, 64),
            epi_tile_mn=(192, 128),
            cluster_shape_mnk=(1, 2, 1),
            atom_layout_mn=(3, 1),
            ab_stage=4,
            epi_stage=1,
            is_persistent=True,
            gemm_n_prologue=0,
            pingpong=False,
        )
    
    if (m, n) == (3072, 3072):
        print('Using specialized config')
        ckernel = RMSNormLinear2SM90(
            tile_shape_mnk=(192, 192, 64),
            epi_tile_mn=(192, 64),
            cluster_shape_mnk=(1, 2, 1),
            atom_layout_mn=(3, 1),
            ab_stage=3,
            epi_stage=2,
            is_persistent=True,
            gemm_n_prologue=0,
            pingpong=False,
        )

    # 2048 1024 16384
    # ckernel = RMSNormLinear2SM90(
    #     tile_shape_mnk=(128, 128, 64),
    #     epi_tile_mn=(128, 32),
    #     cluster_shape_mnk=(1, 2, 1),
    #     atom_layout_mn=(2, 1),
    #     ab_stage=6,
    #     epi_stage=3,
    #     is_persistent=True,
    #     gemm_n_prologue=0,
    #     pingpong=False,
    # )
    
    ckernel_2 = RMSNormLinear1SM90(
        tile_shape_mn=(128, 256), 
        epi_tile_mn=(128, 32),
        cluster_shape_mnk=(2, 1, 1), 
        atom_layout_mn=(2, 1),
        ab_stage=3,
        reuse_ab=False,
        is_persistent=True,
        gemm_n_prologue=0,
        eps=EPS)
    tensors = (a, b, c, EPS)
    compiled_cutedsl = compile_cutedsl(tensors, ckernel, False)
    compiled_cutedsl(*tensors)
    torch.cuda.synchronize()

    if DO_OLD:
        tensors_old = (a, b, c_old)
        compiled_cutedsl_old = compile_cutedsl(tensors_old, ckernel_2, True)
        compiled_cutedsl_old(*tensors_old, STREAM)
        torch.cuda.synchronize()

    if not IS_NCU:
        ref_rmse = get_rmse(ref64, ref.to(htype))
        my_rmse = get_rmse(ref64, c.to(htype))
        my_old_rmse = get_rmse(ref64, c_old.to(htype)) if DO_OLD else 'n/a'
        max_abs_err = (ref64 - c.to(htype)).max().item()
        print(f'{ref_rmse=}, {my_rmse=}, {my_old_rmse=}')
        print(f'{max_abs_err=}')
    
    if IS_SPEED:
        def cdsl_kernel(a_: torch.Tensor, b_: torch.Tensor):
            o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
            compiled_cutedsl(a_, b_, o, EPS)
            return o
        
        def cdsl_kernel_old(a_: torch.Tensor, b_: torch.Tensor):
            o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
            compiled_cutedsl_old(a_, b_, o, STREAM)
            return o
        
        my_ms = do_bench(lambda: cdsl_kernel(a, b))
        time.sleep(2)
        if DO_OLD:
            my_old_ms = do_bench(lambda: cdsl_kernel_old(a, b))
            time.sleep(2)
        else:
            my_old_ms = 'n/a'
        # gemm_ms = do_bench(lambda: a @ b.t())
        gemm_ms = do_bench(lambda: gemm_kernel(a, b))
        time.sleep(2)
        torch_ms = do_bench(lambda: compiled_torch(a, b))
        print(f'{my_ms=}, {my_old_ms=}, {torch_ms=}, {gemm_ms=}')
        print(f'Mine : {my_ms} ({torch_ms / my_ms}x)')
        print(f'Old  : {my_old_ms} ({torch_ms / my_old_ms if DO_OLD else "n/a"}x)')
        print(f'Max  : {gemm_ms} ({torch_ms / gemm_ms})')
        print(f"Gemm : ({gemm_ms / my_ms})")