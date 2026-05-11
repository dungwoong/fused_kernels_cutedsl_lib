from typing import Tuple
import enum

import cutlass
from cutlass import cute, pipeline
from cdsl_helpers import shared, mma, pipeline as my_pipeline, layout as my_layout, store as my_store

# m16nNkK e.g. m16n4096k4096 GEMM
# need to transpose to use WGMMA
# actually have N skinny e.g. m4096n16k4096 
# but have the option to transpose the output
# in the epilogue

@cute.jit
def print0(x):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        if tidx == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.print_tensor(x)
    else:
        if tidx == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.printf(x)

def get_epi_tensor_atom(t: cute.Tensor, epi_smem_layout_staged: cute.ComposedLayout, epi_tile: Tuple[int, int]):
    """
    This only works if you want a single stage
    and your epi SMEM layout has a single stage.
    """
    epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
    epi_tma_tensor_layout = cute.composition(cute.make_identity_layout(t.shape), epi_tile)
    op = cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp(cute.nvgpu.cpasync.ReductionOp.ADD)
    tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        op, t, epi_smem_layout, epi_tma_tensor_layout
    )
    return tma_atom, tma_tensor


class Kernel:
    def __init__(
        self, mnk: Tuple[int, int, int],
        stages: int,
        cluster_m: int,
        k_splits: int,
    ):
        self.dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.m, self.n, self.k = mnk
        self.stages = stages

        self.nconsumer_warps = None
        self.cregs, self.pregs = 232, 40
        self.cluster_size_m = cluster_m
        self.mcast = self.cluster_size_m > 1

        self.k_splits = k_splits
    
    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mO: cute.Tensor):
        sA_layout = shared.get_smem_layout_row_major(self.dtype, self.m, self.k, self.stages)
        sB_layout = shared.get_smem_layout_row_major(self.dtype, self.n, self.k, self.stages)
        sO_layout = shared.get_smem_layout_row_major(self.dtype, self.m, self.n, 1)

        tiled_gemm = mma.get_tiled_mma(self.dtype, True, True, self.acc_dtype, self.m, self.n)

        consumer_wgs = tiled_gemm.size // 128
        self.nconsumer_warps = consumer_wgs * 4

        mA_atom, mA_tensor = shared.get_tma_tensor_and_atom(mA, sA_layout, self.m, self.k)
        mB_atom, mB_tensor = shared.get_tma_tensor_and_atom(mB, sB_layout, self.n, self.k, num_mcast=self.cluster_size_m)
        mO_atom, mO_tensor = get_epi_tensor_atom(mO, sO_layout, (self.m, self.n))

        nclusters_m = cute.ceil_div(mA.shape[0], self.m * self.cluster_size_m)
        # No tile scheduler
        # - (cluster_size_m, nclusters_m * k_splits) as grid dim
        # - clusters cooperate to load the B matrix across the multicast dim
        grid = [self.cluster_size_m, nclusters_m * self.k_splits]
        print(f'{grid=}')

        self.kernel(
            mA_atom, mB_atom, mO_atom,
            mA_tensor, mB_tensor, mO_tensor,
            sA_layout, sB_layout, sO_layout,
            tiled_gemm,
            ).launch(
                grid=grid, 
                block=[(self.nconsumer_warps + 4) * cute.arch.WARP_SIZE], 
                cluster=[self.cluster_size_m, 1, 1])
    
    @cute.kernel
    def kernel(
        self,
        mA_atom: cute.CopyAtom, mB_atom: cute.CopyAtom, mO_atom: cute.CopyAtom,
        mA_tensor: cute.Tensor, mB_tensor: cute.Tensor, mO_tensor: cute.Tensor,
        sA_layout: cute.ComposedLayout, sB_layout: cute.ComposedLayout, sO_layout: cute.ComposedLayout,
        tiled_gemm,
        ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        bidx, bidy, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()

        SharedStorage = type("SS", (), dict())
        items = [
            ('sA_ptr', shared.memrange(self.dtype, sA_layout, 1024)),
            ('sB_ptr', shared.memrange(self.dtype, sB_layout, 1024)),
            ('sO_ptr', shared.memrange(self.dtype, sO_layout, 1024)),
            ('pipe_ptr', cute.struct.MemRange[cutlass.Int64, self.stages * 2]),
        ]
        for k, v in items:
            SharedStorage.__annotations__[k] = v
        
        s_alloc = cutlass.utils.SmemAllocator()
        smem = s_alloc.allocate(cute.struct(SharedStorage))
        sA = shared.smem_get_tensor(smem, 'sA_ptr', sA_layout)
        sB = shared.smem_get_tensor(smem, 'sB_ptr', sB_layout)
        sO = shared.smem_get_tensor(smem, 'sO_ptr', sO_layout)

        a_bytes = cute.size_in_bytes(self.dtype, cute.select(sA_layout, mode=[0, 1]))
        b_bytes = cute.size_in_bytes(self.dtype, cute.select(sB_layout, mode=[0, 1]))
        pipe = my_pipeline.make_tma_pipeline(
            smem.pipe_ptr.data_ptr(),
            self.stages,
            num_consumer_warps=self.nconsumer_warps,
            num_bytes=a_bytes + b_bytes,
            mcast_size=self.cluster_size_m,
            cta_layout_vmnk=cute.make_layout((1, self.cluster_size_m, 1, 1)),
        )

        m_tile = (bidy // self.k_splits) * self.cluster_size_m + bidx
        k_split = (bidy % self.k_splits)
        k_iters = cute.size(mA_tensor, mode=[1]) // self.k
        if (warp_idx < self.nconsumer_warps):
            cute.arch.setmaxregister_increase(self.cregs)
            cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            accumulators = mma.get_acc(tiled_gemm, self.m, self.n, self.acc_dtype)
            accumulate_O = False
            for k in cutlass.range(k_split, k_iters, self.k_splits, unroll=1):
                pipe.consumer_wait(cstate, pipe.consumer_try_wait(cstate))
                mma.accumulating_gemm_ss(tidx, tiled_gemm, sA, sB, accumulators, cstate, cstate, accumulate_O, 0)
                accumulate_O = True
                pipe.consumer_release(cstate)
                cstate.advance()
            acc_16 = cute.make_fragment_like(accumulators, self.dtype)
            acc_16.store(accumulators.load().to(self.dtype))
            my_store.store_acc_stmatrix(acc_16, sO[None, None, 0], tiled_gemm, tidx, self.dtype)
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
            cute.arch.barrier_arrive(barrier_id=0, number_of_threads=(self.nconsumer_warps * cute.arch.WARP_SIZE + cute.arch.WARP_SIZE))
            if warp_idx == 0:
                cute.arch.barrier(barrier_id=0, number_of_threads=(self.nconsumer_warps * cute.arch.WARP_SIZE + cute.arch.WARP_SIZE))
                my_store.tma_store_single(sO[None, None, 0], mO_tensor, self.m, self.n, m_tile, 0, mO_atom)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        
        if (warp_idx >= self.nconsumer_warps):
            cute.arch.setmaxregister_decrease(self.pregs)
            if (warp_idx == self.nconsumer_warps):
                pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
                
                # You have to define it this way for some reason...
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                block_in_cluster_coord_mnk = cute.make_layout((self.cluster_size_m, 1)).get_flat_coord(cta_rank_in_cluster)
                b_mcast_mask = cute.make_layout_image_mask(cute.make_layout((self.cluster_size_m, 1)), block_in_cluster_coord_mnk, mode=0)
                for k in cutlass.range(k_split, k_iters, self.k_splits, unroll=1):
                    pipe.producer_acquire(pstate, pipe.producer_try_acquire(pstate))
                    shared.tma_copy(mA_atom, mA_tensor, sA, self.m, self.k, m_tile, k, pipe, pstate)
                    shared.tma_copy(mB_atom, mB_tensor, sB, self.n, self.k, 0, k, pipe, pstate, bidx, cute.make_layout((self.cluster_size_m, 1)), b_mcast_mask)
                    pstate.advance()
                pipe.producer_tail(pstate)