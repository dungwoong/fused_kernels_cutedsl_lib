import cutlass
from cutlass import cute, pipeline
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

@cute.jit
def mma_epilogue_tma(
    tiled_mma: cute.TiledMma, 
    tma_tensor: cute.Tensor, tma_atom: cute.CopyAtom, 
    shared_tensor: cute.Tensor, 
    accumulators: cute.Tensor, 
    tile_shape_m: int, tile_shape_n: int,
    tile_coord_m: int, tile_coord_n: int, 
    tidx: int, warp_idx: int, acc_dtype, elementwise_fn=lambda x: x):
    """
    Pre-conditions: 
    - shared tensor is n-major
    - shared tensor is rank 3 with stages at the end
    - out_dtype is bf16
    """
    out_dtype = cutlass.BFloat16
    epi_stage = cute.size(shared_tensor, mode=[2])
    
    epilogue_barrier = pipeline.NamedBarrier(barrier_id=int(1), num_threads=tiled_mma.size)

    copy_atom_C = get_stmatrix(False, 4, out_dtype)
    tiled_copy_r2s = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

    gC = cute.local_tile(tma_tensor, (tile_shape_m, tile_shape_n), (tile_coord_m, tile_coord_n))
    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
    tRS_sD = thr_copy_r2s.partition_D(shared_tensor)
    tRS_rAcc = tiled_copy_r2s.retile(accumulators)

    rD_shape = cute.shape(thr_copy_r2s.partition_S(shared_tensor))
    tRS_rD_layout = cute.make_layout(rD_shape[:3])
    tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, acc_dtype)
    size_tRS_rD = cute.size(tRS_rD)

    sepi_for_tma_partition = cute.group_modes(shared_tensor, 0, 2)
    tCgC_for_tma_partition = cute.zipped_divide(gC, (cute.size(shared_tensor, mode=[0]), cute.size(shared_tensor, mode=[1]))) # this just happens to be the right shape
    bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
        tma_atom,
        0,
        cute.make_layout(1),
        sepi_for_tma_partition,
        tCgC_for_tma_partition,
    )

    epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
    epi_tile_shape = tCgC_for_tma_partition.shape[1] # the layout of epi tiles
    epi_tile_layout = cute.make_layout(
        epi_tile_shape, stride=(epi_tile_shape[1], 1)
    )

    for epi_idx in cutlass.range_constexpr(epi_tile_num):
        for epi_v in cutlass.range_constexpr(size_tRS_rD):
            # Take a slice of the accumulators
            tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
        
        # Type conversion
        tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, out_dtype)
        acc_vec = tRS_rD.load()
        tRS_rD_out.store(acc_vec.to(out_dtype))

        epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
        # R2S stmatrix
        cute.copy(
            tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
        )
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )
        epilogue_barrier.arrive_and_wait() # Make sure stmatrix is done

        gmem_coord = epi_tile_layout.get_hier_coord(epi_idx) # e.g. (0, 0) to (7, 0)
        if warp_idx == 0:
            cute.copy(
                tma_atom,
                bSG_sD[(None, epi_buffer)],
                bSG_gD[(None, gmem_coord)],
            )
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(epi_stage - 1, read=True)
        epilogue_barrier.arrive_and_wait() # Don't start next stmatrix yet
