import cutlass
from cutlass import cute

@cute.jit
def make_mma_A_reduction_tensor(tiled_mma: cute.TiledMma, tile_m: int, tile_n: int, dtype: Type[cutlass.Numeric]):
    s2r_r_shape = tiled_mma.partition_shape_A(
            (tile_m, tile_n)
        )
    # mma fragment is ((2, 2, 2), nrows, ncols), so we want (2, nrows) so we can reduce across cols
    layout = cute.make_layout((2, s2r_r_shape[1]))
    acc = cute.make_rmem_tensor(layout, dtype)
    acc.fill(0.0)
    return acc

@cute.jit
def row_sum_square_mixed_types(a: cute.Tensor, acc: cute.Tensor, accum_dtype: Type[cutlass.Numeric]):
    """
    First accumulates rowsum of A in its datatype(e.g. bf16)
    Then casts and increments accumulator
    """
    a_mn = my_layout.make_acc_tensor_mn_view(a, False) # ((2, MMA_M), (2, V, MMA_N), ...) rows cols
    for r in cutlass.range_constexpr(cute.size(acc)):
        tmp = accum_dtype(0.0)
        a_row = a_mn[r, None].load().to(accum_dtype)
        # for i in cutlass.range_constexpr(cute.size(a_row.shape)):
        for i in cutlass.range_constexpr(cute.size(a_row.shape)):
            tmp += (a_row[i] * a_row[i])
        acc[r] += tmp