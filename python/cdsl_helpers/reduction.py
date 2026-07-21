import cutlass
from cutlass import cute
from cutlass.cutlass_dsl import Numeric, dsl_user_op, T
from typing import Type
from . import layout as my_layout
from cutlass._mlir.dialects import nvvm, llvm, arith


@cute.jit
def make_mma_A_reduction_tensor(tiled_mma: cute.TiledMma, tile_m: int, tile_n: int, dtype: Type[cutlass.Numeric]):
    # ZEROS
    s2r_r_shape = tiled_mma.partition_shape_A(
            (tile_m, tile_n)
        )
    # mma fragment is ((2, 2, 2), nrows, ncols), so we want (2, nrows) so we can reduce across cols
    layout = cute.make_layout((2, s2r_r_shape[1]))
    acc = cute.make_rmem_tensor(layout, dtype)
    acc.fill(0.0)
    return acc

@cute.jit
def make_mma_A_ninf_tensor(tiled_mma: cute.TiledMma, tile_m: int, tile_n: int, dtype: Type[cutlass.Numeric]):
    s2r_r_shape = tiled_mma.partition_shape_A(
            (tile_m, tile_n)
        )
    # mma fragment is ((2, 2, 2), nrows, ncols), so we want (2, nrows) so we can reduce across cols
    layout = cute.make_layout((2, s2r_r_shape[1]))
    acc = cute.make_rmem_tensor(layout, dtype)
    acc.fill(float('-inf'))
    return acc

@cute.jit
def row_sum_square_mixed_types(a: cute.Tensor, acc: cute.Tensor, intermediate_accum_dtype: Type[cutlass.Numeric]):
    """
    First accumulates rowsum of A in <intermediate_accum_dtype>
    Then casts and increments accumulator
    """
    a_mn = my_layout.make_acc_tensor_mn_view(a, False) # ((2, MMA_M), (2, V, MMA_N), ...) rows cols
    for r in cutlass.range_constexpr(cute.size(acc)):
        tmp = intermediate_accum_dtype(0.0)
        a_row = a_mn[r, None].load().to(intermediate_accum_dtype)
        # for i in cutlass.range_constexpr(cute.size(a_row.shape)):
        for i in cutlass.range_constexpr(cute.size(a_row.shape)):
            tmp += (a_row[i] * a_row[i])
        acc[r] += tmp

@cute.jit
def row_sum_mixed_types(a: cute.Tensor, acc: cute.Tensor, intermediate_accum_dtype: Type[cutlass.Numeric]):
    a_mn = my_layout.make_acc_tensor_mn_view(a, False) # ((2, MMA_M), (2, V, MMA_N), ...) rows cols
    for r in cutlass.range_constexpr(cute.size(acc)):
        tmp = intermediate_accum_dtype(0.0)
        a_row = a_mn[r, None].load().to(intermediate_accum_dtype)
        # for i in cutlass.range_constexpr(cute.size(a_row.shape)):
        for i in cutlass.range_constexpr(cute.size(a_row.shape)):
            tmp += a_row[i]
        acc[r] += tmp

@dsl_user_op
def fmax(a: float | cutlass.Float32, b: float | cutlass.Float32, c: float | cutlass.Float32 | None = None, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        nvvm.fmax(
            T.f32(),
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            c=cutlass.Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )

@cute.jit
def fmax_reduce(x: cute.TensorSSA, init_val: float | cutlass.Float32 | None = None) -> cutlass.Float32:
    res= cute.make_rmem_tensor(x.shape, cutlass.Float32)
    res.store(x)

    # allocate 4 registers, do 4 maxes at a time and then tree-reduce at the end
    # not sure why they chose a factor of 4, might just be empirically the best
    local_max = [res[0], res[1], res[2], res[3]]
    for i in cutlass.range_constexpr(4, cute.size(x.shape), 4): # start stop step
        local_max[0] = fmax(local_max[0], res[i+0])
        local_max[1] = fmax(local_max[1], res[i+1])
        local_max[2] = fmax(local_max[2], res[i+2])
        local_max[3] = fmax(local_max[3], res[i+3])
    local_max[0] = fmax(local_max[0], local_max[1])
    local_max[2] = fmax(local_max[2], local_max[3])
    local_max[0] = fmax(local_max[0], local_max[2])
    return local_max[0] if const_expr(init_val is None) else fmax(local_max[0], init_val)

@cute.jit
def row_max_f32(a: cute.Tensor, acc: cute.Tensor):
    a_mn = my_layout.make_acc_tensor_mn_view(a, False) # ((2, MMA_M), (2, V, MMA_N), ...) rows cols
    for r in cutlass.range_constexpr(cute.size(acc)):
        a_row = a_mn[r, None].load()
        acc[r] = fmax_reduce(a_row, init_val=acc[r])

@cute.jit
def warp_sum_row_mma_layout(
    val: cute.TensorSSA | cute.Numeric,
    ):
    """
    Assumes MMA acc so each 4 threads holds the same row
    
    This can take in ANY layout, but reduces only across
    groups of 4 threads
    """
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_sum_row_mma_layout(val[i])
        return res.load()
    else:
        for i in cutlass.range_constexpr(2):
            val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val

@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric, # SSA : static single assignment(?)
    op: Callable, 
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE
) -> cute.TensorSSA | cute.Numeric:
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        # this is if you're trying to reduce a whole matrix, we just loop through each element individually and return the result
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        # for a number, we just butterfly reduce this
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val

@cute.jit
def warp_max_row_mma_layout(val: cute.TensorSSA | cute.Numeric):
    return warp_reduce(val, cute.arch.fmax, width=4)