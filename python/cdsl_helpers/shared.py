from typing import Type
import cutlass
from cutlass import cute
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils

# this is the library we'd import, generated code can use any functions from this lib


# TODO experiment to see how column-major works
def get_smem_layout_row_major(
    dtype: Type[cutlass.Numeric],
    rows: int,
    cols: int,
    stages: int,
):
    atom = cute.nvgpu.warpgroup.make_smem_layout_atom(sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, dtype, cols), dtype)
    layout = cute.tile_to_shape(atom, (rows, cols, stages), (0, 1, 2))
    return layout


# def get_smem_struct(fields):
#     """
#     dict is name: type
#     """
#     cls = type("SharedStorage", (), dict())
#     cls.__annotations__ = fields
#     return cute.struct(cls)  # maybe we can cute.struct later

def get_smem_struct():
    return type("SharedStorage", (), dict())

def smem_add_shared_tensor(ss, name_field, dtype, smem_layout, align):
    ss.__annotations__[name_field] = cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(smem_layout)], align]

def smem_add_barrier_array(ss, name_field, stages):
    # Don't forget to multiply stages by 2
    ss.__annotations__[name_field] = cute.struct.Align[cute.struct.MemRange[cutlass.Int64, stages], 16]

def smem_get_tensor(storage, field_name, layout: cute.ComposedLayout | cute.Layout):
    # sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
    if isinstance(layout, cute.ComposedLayout):
        return getattr(storage, field_name).get_tensor(layout.outer, swizzle=layout.inner)
    else:
        return getattr(storage, field_name).get_tensor(layout)

def staged_tensor_sizes(dtype, *layouts):
    """Assumes all layouts are 3D with the last dim as stage"""
    sum_bytes = 0
    for l in layouts:
        assert cute.rank(l) == 3, "need rank-3 layouts for SMEM"
        sum_bytes += cute.size_in_bytes(dtype, cute.slice_(l, (None, None, 0)))
    return sum_bytes


def memrange(dtype, smem_layout, align):
    return cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(smem_layout)], align]


def get_tma_tensor_and_atom(tG, shared_layout, rows, cols, num_mcast=1):
    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp() if num_mcast == 1 else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
    return cute.nvgpu.cpasync.make_tiled_tma_atom(
        op,
        tG,
        cute.select(shared_layout, mode=[0, 1]),
        (rows, cols),
        num_multicast=num_mcast,
    )

def get_tma_epi_tensor_and_atom(tG, shared_layout_staged, rows, cols):
    smem_layout = cute.slice_(shared_layout_staged, (None, None, 0))
    d_cta_v_layout = cute.composition(cute.make_identity_layout(tG.shape), (rows, cols))
    op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
    tma_atom_d, tma_tensor_d = cute.nvgpu.cpasync.make_tiled_tma_atom(
        op, tG, smem_layout, d_cta_v_layout
    )
    return tma_atom_d, tma_tensor_d


# TODO make sure the update works on everything
# def tma_get_copy_fn(
#     atom: cute.CopyAtom,
#     cta_coord: cute.Coord,
#     cta_layout: cute.Layout,
#     src_tensor: cute.Tensor,
#     dst_tensor: cute.Tensor,
#     single_stage: bool=False,
#     **kwargs,
# ):
#     """Returns a callable to perform the G2S copy"""
#     src_is_smem = cutlass.const_expr(isinstance(src_tensor.iterator, cute.Pointer) and src_tensor.memspace == cute.AddressSpace.smem)
#     smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)

#     s, g = cute.nvgpu.cpasync.tma_partition(
#         atom,
#         cta_coord,
#         cta_layout,
#         cute.group_modes(smem_tensor, 0, cute.rank(smem_tensor) - 1),
#         cute.group_modes(gmem_tensor, 0, cute.rank(gmem_tensor) - 2),
#     )
#     src, dst = (s, g) if src_is_smem else (g, s)

#     # TODO might need to fix
#     def copy_tma(src_row, src_col, dst_idx, **kwargs2):
#         cute.copy(atom, src[None, src_row, src_col], dst[None, dst_idx], **kwargs2, **kwargs)

#     return copy_tma, s, g


def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    single_stage: bool=False,
    **kwargs,
):
    """Returns a callable to perform the G2S copy"""
    src_is_smem = cutlass.const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)

    s, g = cute.nvgpu.cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, cute.rank(smem_tensor) - (1 if not single_stage else 0)),
        cute.group_modes(gmem_tensor, 0, cute.rank(gmem_tensor) - (2 if not single_stage else 0)),
    )
    src, dst = (s, g) if src_is_smem else (g, s)

    def copy_tma(src_row, src_col, dst_idx, **kwargs2):
        cute.copy(atom, src[None, src_row, src_col], dst[None, dst_idx], **kwargs2, **kwargs)
    
    def copy_tma_single_stage(**kwargs2):
        cute.copy(atom, src, dst, **kwargs, **kwargs2)
    
    def store_tma(src_idx, dst_row, dst_col, **kwargs2):
        cute.copy(atom, src[src_idx], dst[None, dst_row, dst_col], **kwargs2, **kwargs)
    
    def store_tma_single_stage(**kwargs2):
        cute.copy(atom, src, dst, **kwargs, **kwargs2)
    
    return (copy_tma if not single_stage else copy_tma_single_stage) if not src_is_smem else (store_tma if not single_stage else store_tma_single_stage), s, g


@cute.jit
def get_multicast_info(cluster_layout_shape, mode):
    """Returns mask, cta coord(along mode) and cta layout(along mode)"""
    if cutlass.const_expr(cluster_layout_shape is None or mode < 0 or cluster_layout_shape[mode] == 1):
        return 0, 0, None
    cluster_layout = cute.make_layout(cluster_layout_shape)
    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    block_in_cluster_coord = cluster_layout.get_flat_coord(cta_rank_in_cluster)
    mask = cute.make_layout_image_mask(cluster_layout, block_in_cluster_coord, mode=mode)
    return mask, block_in_cluster_coord[mode], cute.make_layout((cluster_layout_shape[mode],))


# @cute.jit
def tma_copy(
    tma_atom: cute.CopyAtom,
    tma_tensor: cute.Tensor,
    s_tensor: cute.Tensor,
    tile_m: int,
    tile_n: int,
    src_row: int,
    src_col: int,
    pipe: cutlass.pipeline.PipelineAsync,
    state: cutlass.pipeline.PipelineState,
    cta_coord: cute.Coord=0,
    cta_layout: cute.Layout=None,
    mcast_mask: any=0,
):
    if cta_layout is None:
        cta_layout = cute.make_layout((1, 1))
    gT = cute.local_tile(tma_tensor, (tile_m, tile_n), (None, None))
    load, _, _ = tma_get_copy_fn(
        tma_atom,
        cta_coord,
        cta_layout,
        gT,
        s_tensor,
    )
    # return load
    load(src_row, src_col, state.index, tma_bar_ptr=pipe.producer_get_barrier(state), mcast_mask=mcast_mask)

