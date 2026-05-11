import cutlass
from cutlass import cute
from typing import Type
from . import shared

def get_stmatrix(transpose: bool, num_matrices: cutlass.Int32, element_type: Type[cutlass.Numeric]):
    return cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=num_matrices),
        element_type,
    )

def store_acc_stmatrix(acc: cute.Tensor, dst: cute.Tensor, tiled_gemm: cute.TiledMma, tidx: int, element_type: Type[cutlass.Numeric]):
    """
    Store with stmatrix.
    dst should have 2 modes(slice the stages mode)
    """
    copy_atom = get_stmatrix(False, 4, element_type)
    thr_copy_r2s = cute.make_tiled_copy_C(copy_atom, tiled_gemm).get_slice(tidx)
    r2s_s = thr_copy_r2s.partition_D(dst)
    r2s_r = thr_copy_r2s.retile(acc)
    cute.copy(copy_atom, r2s_r, r2s_s)

@cute.jit
def tma_store_single(src: cute.Tensor, dst: cute.Tensor, tile_m: int, tile_n: int, idx_m: int, idx_n: int, s2g_atom: cute.CopyAtom):
    gO = cute.local_tile(dst, (tile_m, tile_n), (idx_m, idx_n))
    store_O, _, _ = shared.tma_get_copy_fn(
        s2g_atom, 0, cute.make_layout(1), src, gO, single_stage=True
    )
    store_O()