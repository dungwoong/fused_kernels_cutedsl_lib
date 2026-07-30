import cutlass
from cutlass import cute
from cutlass import Boolean, const_expr, Int32
from cutlass.cute.nvgpu import warpgroup
from typing import Optional, Type
import cutlass.utils.hopper_helpers as sm90_utils


def get_tiled_mma(ab_dtype: Type[cutlass.Numeric], a_major_k, b_major_k, acc_dtype, tile_m, tile_n, a_in_rs=False):
    assert tile_m % 64 == 0, "tiled_mma tile_m must be a multiple of 64 for now"
    assert tile_n % 8 == 0, "tiled_mma tile_n must be a multiple of 8 for now"
    # if tile shape is none then you just do the entire tile size
    a_mode = cute.nvgpu.warpgroup.OperandMajorMode.K if a_major_k else cute.nvgpu.warpgroup.OperandMajorMode.MN
    b_mode = cute.nvgpu.warpgroup.OperandMajorMode.K if b_major_k else cute.nvgpu.warpgroup.OperandMajorMode.MN
    a_source = cute.nvgpu.warpgroup.OperandSource.RMEM if a_in_rs else cute.nvgpu.warpgroup.OperandSource.SMEM
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        ab_dtype,
        ab_dtype,
        a_mode,
        b_mode,
        acc_dtype,
        # TODO this assumes the MMA atom layout
        atom_layout_mnk=(tile_m // 64, 1, 1),
        tiler_mn=(64, tile_n),
        a_source=a_source
    )
    return tiled_mma


def get_acc(tiled_mma: cute.TiledMma, tile_m: int, tile_n: int, dtype: Type[cutlass.Numeric]):
    thr_mma = tiled_mma.get_slice(0)
    acc_shape = thr_mma.partition_shape_C((tile_m, tile_n))
    acc = cute.make_rmem_tensor(acc_shape, dtype)
    return acc

@cute.jit
def copy_a_wgmma(tidx: cutlass.Int32, tiled_mma: cute.TiledMma, sA: cute.Tensor, tile_m: int, tile_n: int, dtype: Type[cutlass.Numeric]):
    """
    sA should be ONLY a single stage(2D tensor)
    Returns the copy in an mma-ready format
    
    no trans ldmatrix
    """
    copy_atom_A = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            False,
            4,
        ),
        dtype,
    )
    tiled_copy_s2r = cute.make_tiled_copy_A(copy_atom_A, tiled_mma)
    thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
    s2r_sA = thr_copy_s2r.partition_S(sA)
    s2r_r_shape = tiled_mma.partition_shape_A(
        (tile_m, tile_n)
    )
    a_regs_mma = cute.make_rmem_tensor(s2r_r_shape, dtype)
    a_regs = thr_copy_s2r.retile(a_regs_mma)
    cute.copy(tiled_copy_s2r, s2r_sA, a_regs)
    return a_regs_mma


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: cutlass.Constexpr[bool] = False,
    wg_wait: cutlass.Constexpr[int] = 0,
) -> None:
    """
    Should work with SS or RS
    """
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):  # m, k, n_iters
        cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    cute.nvgpu.warpgroup.commit_group()
    if const_expr(wg_wait >= 0):
        cute.nvgpu.warpgroup.wait_group(wg_wait)


@cute.jit
def gemm_zero_init(
    tiled_mma: cute.TiledMma,
    shape: cute.Shape,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
) -> cute.Tensor:
    acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), cutlass.Float32)
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    gemm(tiled_mma, acc, rA, rB, zero_init=True, wg_wait=wg_wait)
    return acc


@cute.jit
def gemm_w_index(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: Boolean,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
) -> None:
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    gemm(tiled_mma, acc, rA, rB, zero_init=zero_init, wg_wait=wg_wait)


@cute.jit
def accumulating_gemm_rs(
    tidx: int,
    tiled_mma: cute.TiledMma,
    rA: cute.Tensor,
    sB: cute.Tensor,
    acc: cute.Tensor,
    b_state: cutlass.pipeline.PipelineState | cutlass.Int32,
    accumulate: bool,
    wg_wait: int = 0,
):
    """
    A should already be loaded and in registers, so no need to index
    B is given the way that gemm_ss is.
    """
    b_idx = b_state
    if cutlass.const_expr(isinstance(b_state, cutlass.pipeline.PipelineState)):
        b_idx = b_state.index
    thr_mma = tiled_mma.get_slice(tidx)
    tSrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
    gemm_w_index(
        tiled_mma,
        acc,
        rA,
        tSrB,
        not accumulate,
        A_idx=None,
        B_idx=b_idx,
        wg_wait=wg_wait,
    )

"""
TODO trying the q in regs for attention does not work for some reason, e.g.
wait(Q)
load q to regs
for k in ...
    single_gemm_rs(q_regs, k)

DIAGNOSIS
- set k matrix to ones(dim=128) and q matrix to zeros
- print result of qk every iteration, and compare to gemm_ss results

The output is always the same first iter, then next iter every output is 128.
This suggests that rQ was overwritten by ones, which would come from the K matrix.

Moving rQ inside the k loop fixes things. Printing rQ outside/inside the k-loop also fixes things.
I'm not sure why rQ gets overwritten, but that suggests that the problem may not be with my code,
but rather the CuteDSL/PTXAS compiler?

Probably not a fence or sync problem since a) results are consistent everytime, b) nonsense values are being written into rQ
"""
@cute.jit
def single_gemm_rs(
    tidx: int,
    rows: int,
    cols: int,
    tiled_mma: cute.TiledMma,
    rA: cute.Tensor,
    sB: cute.Tensor,
    b_state: cutlass.pipeline.PipelineState | cutlass.Int32,
    wg_wait: int = 0,
):
    """
    A should already be loaded and in registers, so no need to index
    B is given the way that gemm_ss is.
    """
    b_idx = b_state
    if cutlass.const_expr(isinstance(b_state, cutlass.pipeline.PipelineState)):
        b_idx = b_state.index
    thr_mma = tiled_mma.get_slice(tidx)
    tSrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
    return gemm_zero_init(tiled_mma, (rows, cols), rA, tSrB, A_idx=None, B_idx=b_idx, wg_wait=wg_wait)


@cute.jit
def accumulating_gemm_ss(
    tidx: int,
    tiled_mma: cute.TiledMma,
    sA: cute.Tensor,
    sB: cute.Tensor,
    acc: cute.Tensor,
    a_state: cutlass.pipeline.PipelineState | cutlass.Int32,
    b_state: cutlass.pipeline.PipelineState | cutlass.Int32,
    accumulate: bool,
    wg_wait: int = 0,
):
    a_idx = a_state
    b_idx = b_state
    if cutlass.const_expr(isinstance(a_state, cutlass.pipeline.PipelineState)):
        a_idx = a_state.index
    if cutlass.const_expr(isinstance(b_state, cutlass.pipeline.PipelineState)):
        b_idx = b_state.index
    thr_mma = tiled_mma.get_slice(tidx)
    tSrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
    tSrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
    gemm_w_index(
        tiled_mma,
        acc,
        tSrA,
        tSrB,
        not accumulate,
        A_idx=a_idx,
        B_idx=b_idx,
        wg_wait=wg_wait,
    )


@cute.jit
def single_gemm_ss(
    tidx: int,
    rows: int,
    cols: int,
    tiled_mma: cute.TiledMma,
    sA: cute.Tensor,
    sB: cute.Tensor,
    a_state: cutlass.pipeline.PipelineState | cutlass.Int32,
    b_state: cutlass.pipeline.PipelineState | cutlass.Int32,
    wg_wait: int = 0,
):
    a_idx = a_state
    b_idx = b_state
    if cutlass.const_expr(isinstance(a_state, cutlass.pipeline.PipelineState)):
        a_idx = a_state.index
    if cutlass.const_expr(isinstance(b_state, cutlass.pipeline.PipelineState)):
        b_idx = b_state.index
    thr_mma = tiled_mma.get_slice(tidx)
    tSrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
    tSrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
    return gemm_zero_init(tiled_mma, (rows, cols), tSrA, tSrB, A_idx=a_idx, B_idx=b_idx, wg_wait=wg_wait)
