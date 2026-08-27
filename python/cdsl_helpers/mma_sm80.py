import cutlass
from cutlass import cute
from typing import Optional, Type


# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma
# FP16 supports many things, but bf16 may only support 16 8 16
def get_tiled_mma(
    atom_layout,
    ab_dtype: Type[cutlass.Numeric]=cutlass.BFloat16,
    acc_dtype: Type[cutlass.Numeric]=cutlass.Float32,
    mnk=(16, 8, 16)):
    """
    Both A and B must be k-major for this, and must be in registers.
    """
    op = cute.nvgpu.warp.MmaF16BF16Op(
        ab_dtype,
        acc_dtype,
        mnk,
    )
    tC = cute.make_layout(atom_layout)
    permutation_mnk = (
        atom_layout[0] * mnk[0],
        atom_layout[1] * mnk[1] * 2,
        atom_layout[2] * mnk[2],
    )
    return cute.make_tiled_mma(
        op, tC, permutation_mnk
    )

@cute.jit
def copy_mma_bf16(tidx: cutlass.Int32, tiled_mma: cute.TiledMma, sX: cute.Tensor, is_a: cutlass.Constexpr[bool]):
    """
    Need k major
    """
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
        cutlass.BFloat16
    )
    tiled_copy_fn = cute.make_tiled_copy_A if cutlass.const_expr(is_a) else cute.make_tiled_copy_B
    tiled_copy = tiled_copy_fn(copy_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(tidx)
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA_view = thr_copy.partition_S(sX)
    tCsA = thr_mma.partition_A(sX) if cutlass.const_expr(is_a) else thr_mma.partition_B(sX)
    tCrA = tiled_mma.make_fragment_A(tCsA) if cutlass.const_expr(is_a) else tiled_mma.make_fragment_B(tCsA)
    tCrA_copy_view = thr_copy.retile(tCrA)
    # I THINK THIS WORKS NOW, GEMM IS ERRORING NOW
    # print('sX', sX)
    # print('tCrA', tCrA)
    # print('tCsA_view', tCsA_view)
    # print('tCrA_copy_view', tCrA_copy_view)
    for k in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
        cute.copy(tiled_copy, tCsA_view[None, None, k], tCrA_copy_view[None, None, k])
    return tCrA

# mma.get_acc SHOULD work for sm80 mmas too, except we must fill with 0
# accumulators.fill(0.0)
def get_acc(tiled_mma: cute.TiledMma, dtype: Type[cutlass.Numeric]=cutlass.Float32):
    thr_mma = tiled_mma.get_slice(0)
    acc_shape = thr_mma.partition_shape_C((tile_m, tile_n))
    acc = cute.make_rmem_tensor(acc_shape, dtype)
    return acc

@cute.jit
def gemm_0(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
) -> None:
    """
    tCrA, tCrB in registers, both k-major.
    """
    for k in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)