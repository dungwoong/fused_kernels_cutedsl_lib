import argparse
from typing import Callable, Tuple, Type
import math
import cuda.bindings.driver as cuda

import torch
from triton import runtime
import functools
import statistics

import cutlass
from cutlass import Boolean, Int32, const_expr
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait, PipelineState, PipelineUserType
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
from cutedsl_kernels import RMSNormLinear1SM90

if __name__ == "__main__":
    print('Starting...')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["debug", "speed", "ncu"])
    args = parser.parse_args()
    IS_NCU = args.mode == 'ncu'
    IS_DEBUG = args.mode == 'debug'
    IS_SPEED = args.mode == 'speed'

    m, n, k = 4096, 4096, 4096
    flops = 2 * m * n * k

    def get_tflops(time_ms):
        return (flops / (time_ms / 1e3)) / 1e12

    a = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    norm_a = torch.nn.functional.rms_norm(a.to(torch.float32), normalized_shape=(k,), eps=1e-5)
    b = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    ref = (norm_a @ b.t().to(torch.float32)).to(torch.bfloat16)
    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )
    a_cute, b_cute, c_cute = [convert_from_dlpack(x) for x in (a, b, c)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # 0.1803, 762TFLOPs(gemm_cuteDSL) to 0.1847, 744 TFLOPs(load to registers first)
    # for some reason I'm at 752 after adding the reduction but idk

    # RMS fusion takes 0.1921ms whereas compiled torch is 0.2006ms
    gemm = RMSNormLinear1SM90(tile_shape_mn=(128, 256), 
                    epi_tile_mn=(128, 32),
                    cluster_shape_mnk=(2, 1, 1), 
                    atom_layout_mn=(2, 1),
                    ab_stage=3,
                    reuse_ab=False,
                    is_persistent=True,
                    gemm_n_prologue=0)
    compiled_gemm = cute.compile(gemm, a_cute, b_cute, c_cute, current_stream)
    compiled_gemm(a_cute, b_cute, c_cute, current_stream)
    print('All close:', torch.allclose(ref, c, atol=1e-1, rtol=1e-1))
    if IS_DEBUG:
        print(ref)
        print(c)
        print((ref - c)[0, :64])

    if IS_DEBUG:
        n_incorrect = c.numel() - ((c - ref).abs() < 1).sum()
        print('max_incorrect :', torch.max((c - ref).abs()).item())
        print('max_rel_incorrect :', torch.max(((c - ref).abs() / ref.abs().clamp_min(1e-8))).item())
        print('n_incorrect :', n_incorrect)
        print('n_nonzero :', (c != 0).sum())

    def profile_ms(op, repeats=30):

        clear_cache = functools.partial(
            runtime.driver.active.clear_cache,  # type: ignore[attr-defined]
            runtime.driver.active.get_empty_cache_for_benchmark(),  # type: ignore[attr-defined]
        )
        clear_cache()

        # warmup
        op()
        torch.cuda.synchronize()

        start = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

        for i in range(repeats):
            clear_cache()
            start[i].record()
            op()
            end[i].record()

        torch.cuda.synchronize()
        return statistics.median([s.elapsed_time(e) for s, e in zip(start, end)])

    @torch.compile
    def torch_gemm():
        a_rms = torch.nn.functional.rms_norm(a, normalized_shape=(k,), eps=1e-5)
        return a_rms @ b.t()

    if IS_SPEED:
        my_ms = profile_ms(lambda: compiled_gemm(a_cute, b_cute, c_cute, current_stream))
        other_ms = profile_ms(torch_gemm)
        print(f'{my_ms=}, {other_ms=}')
        my_flops, other_flops = get_tflops(my_ms), get_tflops(other_ms)
        print(f'{my_flops=}, {other_flops=}')