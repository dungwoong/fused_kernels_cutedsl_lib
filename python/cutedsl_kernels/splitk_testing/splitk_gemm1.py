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
    # print('id:', cute.make_identity_layout(t.shape))
    epi_tma_tensor_layout = cute.composition(cute.select(cute.make_identity_layout(t.shape), mode=[0, 1]), epi_tile)
    # epi_tma_tensor_layout = cute.select(epi_tma_tensor_layout, mode=[2, 0, 1])
    # print('epi tma tensor', epi_tma_tensor_layout)
    op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
    tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        op, t, epi_smem_layout, epi_tma_tensor_layout, epi_tile
    )
    # print('tma_tensor', tma_tensor)
    return tma_atom, tma_tensor

def store_acc_fp32(accumulator: cute.Tensor, shared_tensor: cute.Tensor, tiled_gemm: cute.TiledMma, tidx):
    thr_mma = tiled_gemm.get_slice(tidx)
    tCgC = thr_mma.partition_C(shared_tensor)
    cute.autovec_copy(accumulator, tCgC)



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
        # mO is [row, col, splits] but torch shape is [row, splits, col]
        # I'm pretty sure this above comment is outdated now.
        mO = my_layout.select(mO, [0, 2, 1])
        sA_layout = shared.get_smem_layout_row_major(self.dtype, self.m, self.k, self.stages)
        sB_layout = shared.get_smem_layout_row_major(self.dtype, self.n, self.k, self.stages)
        sO_layout = shared.get_smem_layout_row_major(self.acc_dtype, self.m, self.n, 1)

        tiled_gemm = mma.get_tiled_mma(self.dtype, True, True, self.acc_dtype, self.m, self.n)

        consumer_wgs = tiled_gemm.size // 128
        self.nconsumer_warps = consumer_wgs * 4

        mA_atom, mA_tensor = shared.get_tma_tensor_and_atom(mA, sA_layout, self.m, self.k)
        mB_atom, mB_tensor = shared.get_tma_tensor_and_atom(mB, sB_layout, self.n, self.k, num_mcast=self.cluster_size_m)
        mO_atom, mO_tensor = get_epi_tensor_atom(mO, sO_layout, (self.m, self.n))
        # print('mO_tensor', mO_tensor)

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
            ('sO_ptr', shared.memrange(self.acc_dtype, sO_layout, 1024)),
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
            # acc_16 = cute.make_fragment_like(accumulators, self.dtype)
            # acc_16.store(accumulators.load().to(self.dtype))
            
            # TODO
            # my_store.store_acc_stmatrix(acc_16, sO[None, None, 0], tiled_gemm, tidx, self.dtype)
            store_acc_fp32(accumulators, sO[None, None, 0], tiled_gemm, tidx)
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
            cute.arch.barrier_arrive(barrier_id=0, number_of_threads=(self.nconsumer_warps * cute.arch.WARP_SIZE + cute.arch.WARP_SIZE))
            if warp_idx == 0:
                # print0(sO[None, None, 0])
                cute.arch.barrier(barrier_id=0, number_of_threads=(self.nconsumer_warps * cute.arch.WARP_SIZE + cute.arch.WARP_SIZE))
                # print0(mO_tensor[None, None, k_split])
                my_store.tma_store_single(sO[None, None, 0], mO_tensor[None, None, k_split], self.m, self.n, m_tile, 0, mO_atom)
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


class ReduceDowncastKernel:
    """
    input is (m, splits, n) in FP32, reduces to (m, n) in BF16

    Thread will do <tile_n>, accumulating all splits and storing.
    The CTA will do <tile_m, tile_n> altogether

    Copy takes e.g. mA[None, 0, None] and copy some of that out. Then take mA[None, 1, None] etc.
    """
    def __init__(
        self,
        tile_m, tile_n,
        splits: int,
    ):
        """
        The actual tile will be (tile_m * splits, tile_n)
        """
        self.m, self.n = tile_m, tile_n
        self.splits = splits
        self.dtype = cutlass.Float32
        self.out_dtype = cutlass.BFloat16
    
    @cute.jit
    def __call__(self, mA: cute.Tensor, mO: cute.Tensor):
        # mA is (m, splits, n)
        # mO is just (m, n)
        # so you can divide (m * n) among threads
        # so we need to load in (mTile, splits, n)
        grid = (cute.size(mA, mode=[0]) // self.m, cute.size(mA, mode=[2]) // self.n)
        # mA = cute.group_modes(mA, 1, 3)
        copy_in = self.get_tiled_copy()
        # print(copy_in)
        store_out = self.get_tiled_store()
        num_threads = copy_in.size
        print('reduce grid', grid)
        print('reduce block', num_threads)

        self.kernel(mA, mO, copy_in, store_out).launch(grid=grid, block=[num_threads, 1, 1])

    @cute.kernel
    def kernel(self, mA: cute.Tensor, mO: cute.Tensor, copy_in: cute.TiledCopy, store_out: cute.TiledCopy):
        # mA is already combined
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        acc = cute.make_rmem_tensor((1, self.n), self.out_dtype)
        dst = cute.make_rmem_tensor((1, self.n, self.splits), self.dtype)

        # print(cute.rank(mA_combined))
        thr_copy = copy_in.get_slice(tidx)
        for i in cutlass.range_constexpr(self.splits):
            mA_slice = mA[None, i, None]
            gA = cute.local_tile(mA_slice, (self.m, self.n), (bidx, bidy))
            tAgA = thr_copy.partition_S(gA)
            dst_copy = thr_copy.partition_D(dst)
            # print('dst_copy', dst_copy)

            cute.copy(copy_in, tAgA[None, None, 0], dst_copy[None, None, 0, i])
        # print0(dst)

        for i in cutlass.range_constexpr(cute.size(acc)):
            tmp = self.dtype(0)
            for j in cutlass.range_constexpr(i, self.n * self.splits, self.n):
                tmp += dst[j]
            acc[i] = self.out_dtype(tmp)
        # print0(acc)
        
        thr_store = store_out.get_slice(tidx)
        gO = cute.local_tile(mO, (self.m, self.n), (bidx, bidy))
        tOgO = thr_store.partition_D(gO)
        acc_ret = thr_store.partition_S(acc)
        cute.copy(store_out, acc_ret, tOgO)
        
    def get_tiled_copy(self):
        """
        threads hold <splits, nTile> items
        """
        copy_op = cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(copy_op, self.dtype, num_bits_per_copy=128)
        tiler_mn = (self.m, self.n * self.splits)
        layout_tv = cute.make_layout((self.m, (self.n)), stride=(1, (self.m)))
        return cute.make_tiled_copy(copy_atom, layout_tv, tiler_mn)
        # return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    
    def get_tiled_store(self):
        # we loaded in mTile x (splits * nTile)
        # we store out mTile x nTile
        copy_op = cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(copy_op, self.out_dtype, num_bits_per_copy=128)
        tiler_mn = (self.m, self.n)
        layout_tv = cute.make_layout((self.m, self.n), stride=(1, self.m))
        return cute.make_tiled_copy(copy_atom, layout_tv, tiler_mn)