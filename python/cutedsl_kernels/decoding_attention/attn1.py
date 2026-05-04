from typing import Tuple
import enum
import math
import operator

import cutlass
from cutlass import cute, pipeline
from cdsl_helpers import shared, mma, pipeline as my_pipeline, layout as my_layout, store as my_store
from . import attn_scheduler, my_utils

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

class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()
    WG_Sync = enum.auto()
    Reduction_Sync = enum.auto()


def get_colsum_init_16(dtype):
    """
    To colsum a 16x16 block from a ldmatrix formulation, your thread holds 2 items in each 8x8 grid
    e.g. thread 0 holds (0, 1) and (8, 9)
    """
    return cute.make_rmem_tensor(4, dtype)

@cute.jit
def colsum_16(t: cute.Tensor, acc: cute.Tensor, is_first: bool):
    """
    Sum over columns of an MMA accumulator, outputting into <acc>
    """
    assert cute.size(t, mode=[1]) == 1 and cute.size(t, mode=[2]) == 1, t # should be ((2, 2, 2), 1, 1)
    t_mn = my_layout.make_acc_tensor_mn_view(t) # ((2, 2, V), MMA_M, MMA_N, ...) > ((2, MMA_M), (2, V, MMA_N), ...), we want to sum over the first mode
    for c in cutlass.range(cute.size(acc), unroll_full=True):
        t_col = t_mn[None, c].load()
        new_sum = my_utils.fadd_reduce(t_col, init_val=acc[c] if cutlass.const_expr(not is_first) else None)
        acc[c] = new_sum


@cute.jit
def warp_sum_column(
    val: cute.TensorSSA | cute.Numeric,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE
    ):
    """
    Assumes MMA acc so each 4 threads holds the same column
    (first row is 0-3, second is 4-7)
    """
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_sum_column(val[i], width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(2, int(math.log2(width))):
            val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val

@cute.jit
def block_col_sum(
    acc: cute.TensorSSA,
    reduction_buffer: cute.Tensor,
    num_threads: cutlass.Int32,
    dtype
    ):
    """
    Buffer will be (16, num_warps)
    threads 0-3 save 0-7
    threads 4-7 save 8-15
    Threads are holding a (4,) accumulator for colsum
    """
    nwarps = cute.size(reduction_buffer.shape[1])
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    idx0, idx1 = lane_idx // 4, lane_idx % 4
    st_idx = idx0 * 8 + idx1 * 2
    reg_idx = idx0 * 2
    if lane_idx < 8:
        reduction_buffer[st_idx, warp_idx] = acc[reg_idx]
        reduction_buffer[st_idx + 1, warp_idx] = acc[reg_idx + 1]
    
    # For now, just do 16 warp reductions. In the future, I can split up the work
    # among threads in the warp and even different warps in the WG
    cute.arch.barrier(barrier_id=NamedBarrierFwd.Reduction_Sync, number_of_threads=num_threads)
    out = cute.make_rmem_tensor(16, dtype)
    for i in cutlass.range(16, unroll_full=True):
        block_reduce_val = 0.0
        if lane_idx < nwarps:
            block_reduce_val = reduction_buffer[i, lane_idx]
        out[i] = cute.arch.warp_reduction(block_reduce_val, operator.add)
    return out

@cute.jit
def grab_values_16(reduce_acc: cute.Tensor, dtype):
    output = cute.make_rmem_tensor(4, dtype)
    lane_idx = cute.arch.lane_idx()
    idx0 = (lane_idx % 4) * 2 # NOTE missed the *2 initially
    output[0] = reduce_acc[idx0]
    output[1] = reduce_acc[idx0 + 1]
    output[2] = reduce_acc[8 + idx0]
    output[3] = reduce_acc[8 + idx0 + 1]
    return output

@cute.jit
def scale_exp(acc: cute.Tensor, scale_log2: cutlass.Float32):
    out = my_utils.exp2f(acc.load() * scale_log2)
    acc.store(out)

@cute.jit
def col_scale(gemm_acc: cute.Tensor, scale: cute.Tensor):
    """
    We expect scale to be a (4,) tensor
    and the gemm_acc is from wgmma
    """
    acc_mn = my_layout.make_acc_tensor_mn_view(gemm_acc) # ((2, MMA_M), (2, V, MMA_N), ...)
    for c in cutlass.range(cute.size(scale), unroll_full=True):
        acc_col = acc_mn[None, c].load() * cute.arch.rcp_approx(scale[c])
        acc_mn[None, c].store(acc_col)



def get_epi_tensor_atom(t: cute.Tensor, epi_smem_layout_staged: cute.ComposedLayout, epi_tile: Tuple[int, int]):
    """
    This only works if you want a single stage
    and your epi SMEM layout has a single stage.
    """
    epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
    epi_tma_tensor_layout = cute.composition(cute.make_identity_layout(t.shape), epi_tile)
    op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
    tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        op, t, epi_smem_layout, epi_tma_tensor_layout
    )
    return tma_atom, tma_tensor

def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))

@cute.jit
def _store_t(src: cute.Tensor, dst: cute.Tensor, tiled_gemm: cute.TiledMma, tidx: int, element_type):
    dst_t = transpose_view(dst)
    copy_atom = my_store.get_stmatrix(True, 4, element_type)
    thr_copy_r2s = cute.make_tiled_copy_C(copy_atom, tiled_gemm).get_slice(tidx)
    r2s_s = thr_copy_r2s.partition_D(dst_t)
    r2s_r = thr_copy_r2s.retile(src)
    cute.copy(copy_atom, r2s_r, r2s_s)

@cute.jit
def _tma_store_single(src: cute.Tensor, dst: cute.Tensor, tile_m: int, tile_n: int, idx_m: int, idx_n: int, s2g_atom: cute.CopyAtom):
    gO = cute.local_tile(dst, (tile_m, tile_n), (idx_m, idx_n))
    store_O, _, _ = shared.tma_get_copy_fn(
        s2g_atom, 0, cute.make_layout(1), src, gO, single_stage=True
    )
    store_O()

class Kernel:
    """
    Assume:
    - dim_q = dim_v so qk_mnk defines the entire operation
    - no multicasting. The triton kernel didn't have it either.
    
    - see if I can get consumer regs down to 160, then I can get a 3rd warpgroup
    """
    def __init__(
        self,
        qk_mnk: Tuple[int, int, int],
        stages: int,
        p_stages: int,
        is_persistent: bool
    ):
        self.dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.seq_q, self.tile_k, self.dim = qk_mnk
        self.stages = stages
        self.p_stages = p_stages

        self.consumer_regs = 232
        self.producer_regs = 40
        self.nconsumer_warps = None

        self.is_persistent = is_persistent
    
    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor, softmax_scale: cutlass.Float32):
        """
        e.g. mQ is (nheads, 16, 128)
             mK, mV are (nheads, 1024, 128)
             mO is (nheads, 16, 128)
        """
        # seqlen, dim, heads
        mQ = my_layout.select(mQ, [1, 2, 0])
        mK = my_layout.select(mK, [1, 2, 0])
        mV = my_layout.select(mV, [1, 2, 0])
        mO = my_layout.select(mO, [1, 2, 0])

        qk_gemm = mma.get_tiled_mma(self.dtype, True, True, self.acc_dtype, self.tile_k, self.seq_q)
        pv_gemm = mma.get_tiled_mma(self.dtype, True, True, self.acc_dtype, self.dim, self.seq_q)
        assert qk_gemm.size == pv_gemm.size
        consumer_wgs = qk_gemm.size // 128
        self.nconsumer_warps = consumer_wgs * 4

        # QKt = m16nKk128, KQt = mKn16k128 = Pt --> (k, 16)
        sQ_layout = shared.get_smem_layout_row_major(self.dtype, self.seq_q, self.dim, 1)
        sK_layout = shared.get_smem_layout_row_major(self.dtype, self.tile_k, self.dim, self.stages)
        
        # PV = m16n128kK, VtPt = m128n16kK
        # NOTE this requires V to be stored in a transposed fashion, which is alright for inference
        # but WGMMA should support A as mn-major according to PTX ISA...
        sP_layout = shared.get_smem_layout_row_major(self.dtype, self.seq_q, self.tile_k, self.p_stages)
        sV_layout = shared.get_smem_layout_row_major(self.dtype, self.dim, self.tile_k, self.stages)
        sO_layout = shared.get_smem_layout_row_major(self.dtype, self.seq_q, self.dim, 1)
        sSum_layout = cute.make_layout((self.seq_q, self.nconsumer_warps)) # length, num_warps

        mQ_g2s_atom, mQ_g2s_tensor = shared.get_tma_tensor_and_atom(mQ, sQ_layout, self.seq_q, self.dim)
        mK_g2s_atom, mK_g2s_tensor = shared.get_tma_tensor_and_atom(mK, sK_layout, self.tile_k, self.dim)
        mV_g2s_atom, mV_g2s_tensor = shared.get_tma_tensor_and_atom(mV, sV_layout, self.dim, self.tile_k)
        mO_s2g_atom, mO_s2g_tensor = get_epi_tensor_atom(mO, sO_layout, (self.seq_q, self.dim))

        nheads = mK.shape[2]
        scheduler_params = attn_scheduler.HeadAttnTileScheduler.to_underlying_arguments(
            attn_scheduler.HeadAttnTileSchedulerArguments.create(nheads, self.is_persistent)
        )
        grid = attn_scheduler.HeadAttnTileScheduler.get_grid_shape(scheduler_params, 132)
        # KERNEL LAUNCH
        softmax_scale_log2 = softmax_scale * math.log2(math.e)

        self.kernel(
            scheduler_params, softmax_scale_log2,
            sQ_layout, sK_layout, sV_layout, sP_layout, sO_layout, sSum_layout,
            mQ_g2s_atom, mQ_g2s_tensor,
            mK_g2s_atom, mK_g2s_tensor,
            mV_g2s_atom, mV_g2s_tensor,
            mO_s2g_atom, mO_s2g_tensor,
            qk_gemm, pv_gemm,
        ).launch(grid=grid, block=[(self.nconsumer_warps + 4) * cute.arch.WARP_SIZE])
    
    @cute.kernel
    def kernel(
        self,
        scheduler_params, softmax_scale_log2,
        sQ_layout, sK_layout, sV_layout, sP_layout, sO_layout, sSum_layout,
        mQ_g2s_atom, mQ, # TMA tensor
        mK_g2s_atom, mK,
        mV_g2s_atom, mV,
        mO_s2g_atom, mO,
        qk_gemm, pv_gemm,
        ):
        """
        load q
        for k:
            load k
            p.t = mma(k, q.t) # (K, 128) x (128, 16) --> (K, 16)
            p.t = exp(p.t/d)

            # We also need 16 row-sums for this.
            # what if for the first iter, I omit this and see how it goes

            store_t(p) # (16, K)
            load v
            o = mma(v.t, p.t) # (128, K) x (16, K) --> (128, 16)
        epilogue
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        ss_t, bars_t = self._shared_cls(sQ_layout, sK_layout, sV_layout, sP_layout, sO_layout, sSum_layout)
        s_alloc = cutlass.utils.SmemAllocator()
        dsmem = s_alloc.allocate(ss_t)
        smem_bars = s_alloc.allocate(bars_t)

        sQ = shared.smem_get_tensor(dsmem, 'sQ_ptr', sQ_layout)
        sK = shared.smem_get_tensor(dsmem, 'sK_ptr', sK_layout)
        sV = shared.smem_get_tensor(dsmem, 'sV_ptr', sV_layout)
        sP = shared.smem_get_tensor(dsmem, 'sP_ptr', sP_layout)
        sO = shared.smem_get_tensor(dsmem, 'sO_ptr', sO_layout)
        sSum = shared.smem_get_tensor(dsmem, 'sum_buf', sSum_layout)

        q_bytes = cute.size_in_bytes(self.dtype, cute.select(sQ_layout, mode=[0, 1]))
        k_bytes = cute.size_in_bytes(self.dtype, cute.select(sK_layout, mode=[0, 1]))
        v_bytes = cute.size_in_bytes(self.dtype, cute.select(sV_layout, mode=[0, 1]))
        pipe_k = my_pipeline.make_tma_pipeline(
            smem_bars.k_pipe_ptr.data_ptr(),
            self.stages,
            num_consumer_warps=self.nconsumer_warps,
            num_bytes=k_bytes,
            mcast_size=1, # TODO add multicasting later
            cta_layout_vmnk=None
        )
        pipe_v = my_pipeline.make_tma_pipeline(
            smem_bars.v_pipe_ptr.data_ptr(),
            self.stages,
            num_consumer_warps=self.nconsumer_warps,
            num_bytes=v_bytes,
            mcast_size=1,
            cta_layout_vmnk=None
        )
        scheduler = attn_scheduler.HeadAttnTileScheduler.create(scheduler_params)
        k_iters = cute.size(mK, mode=[0]) // self.tile_k
        if (warp_idx < self.nconsumer_warps): # Consumer
            cute.arch.setmaxregister_increase(self.consumer_regs)
            state_k = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            state_v = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            
            acc_o = mma.get_acc(pv_gemm, self.dim, self.seq_q, self.acc_dtype)
            work_tile = scheduler.initial_work_tile_info()
            if work_tile.is_valid_tile:
                tile_coord = work_tile.tile_idx
                head_idx = tile_coord[0]

                acc_sum = get_colsum_init_16(self.acc_dtype)
                acc_sum.fill(0.0) # NOTE we could do is_first but this is easier
                accumulate_O = False
                for k in cutlass.range(k_iters, unroll=1):
                    pipe_k.consumer_wait(state_k, pipe_k.consumer_try_wait(state_k))
                    acc_p = mma.single_gemm_ss(tidx, self.tile_k, self.seq_q, qk_gemm, sK, sQ, state_k, 0)
                    pipe_k.consumer_release(state_k)

                    scale_exp(acc_p, softmax_scale_log2)
                    
                    # Do the processing, and sum potentially
                    colsum_16(acc_p, acc_sum, is_first=False)

                    acc_p_16 = cute.make_fragment_like(acc_p, self.dtype)
                    acc_p_16.store(acc_p.load().to(self.dtype))
                    _store_t(acc_p_16, sP[None, None, 0], qk_gemm, tidx, self.dtype)
                    cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
                    cute.arch.barrier(barrier_id=NamedBarrierFwd.WG_Sync, number_of_threads=(self.nconsumer_warps * cute.arch.WARP_SIZE))
                    
                    # sP is (16, 128) P, I verified
                    pipe_v.consumer_wait(state_v, pipe_v.consumer_try_wait(state_v))

                    # this should be Vt Pt = (PV)t
                    mma.accumulating_gemm_ss(tidx, pv_gemm, sV, sP, acc_o, state_v, 0, accumulate_O, 0)
                    accumulate_O = True
                    pipe_v.consumer_release(state_v)
                    state_k.advance()
                    state_v.advance()
                
                # print0(acc_sum)
                warp_sum = warp_sum_column(acc_sum.load())
                # print0(warp_sum)
                col_sum = block_col_sum(warp_sum, sSum, self.nconsumer_warps * cute.arch.WARP_SIZE, self.acc_dtype)
                # print0(col_sum)

                # TODO rescale acc here
                col_sum_needed = grab_values_16(col_sum, self.acc_dtype) # (4,) tensor for the sum
                col_scale(acc_o, col_sum_needed)

                acc_o_16 = cute.make_fragment_like(acc_o, self.dtype)
                acc_o_16.store(acc_o.load().to(self.dtype))

                # seq_q, dim
                _store_t(acc_o_16, sO[None, None, 0], pv_gemm, tidx, self.dtype)
                cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
                cute.arch.barrier_arrive(barrier_id=NamedBarrierFwd.Epilogue, number_of_threads=(self.nconsumer_warps + 1) * cute.arch.WARP_SIZE)
                # print0(sP[None, None, 0])
                if warp_idx == 0:
                    cute.arch.barrier(barrier_id=NamedBarrierFwd.Epilogue, number_of_threads=(self.nconsumer_warps + 1) * cute.arch.WARP_SIZE)
                    curr_o = mO[None, None, head_idx]
                    _tma_store_single(sO[None, None, 0], curr_o, self.seq_q, self.dim, 0, 0, mO_s2g_atom)
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

        if (warp_idx >= self.nconsumer_warps): # Producer
            cute.arch.setmaxregister_decrease(self.producer_regs)
            if (warp_idx == self.nconsumer_warps):
                state_k = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
                state_v = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)

                work_tile = scheduler.initial_work_tile_info()
                if work_tile.is_valid_tile:
                    tile_coord = work_tile.tile_idx
                    head_idx = tile_coord[0]
                    q_curr = mQ[None, None, head_idx]
                    k_curr = mK[None, None, head_idx] # seqlen, dim
                    v_curr = mV[None, None, head_idx]
                    
                    # Load Q and first K
                    pipe_k.producer_acquire(state_k, pipe_k.producer_try_acquire(state_k), extra_tx_count=q_bytes)
                    # load q
                    shared.tma_copy(mQ_g2s_atom, q_curr, sQ, self.seq_q, self.dim, 0, 0, pipe_k, state_k)
                    # load k[0]
                    shared.tma_copy(mK_g2s_atom, k_curr, sK, self.tile_k, self.dim, 0, 0, pipe_k, state_k)
                    # load v[0]
                    pipe_v.producer_acquire(state_v, pipe_v.producer_try_acquire(state_v))
                    shared.tma_copy(mV_g2s_atom, v_curr, sV, self.dim, self.tile_k, 0, 0, pipe_v, state_v)
                    state_k.advance()
                    state_v.advance()
                    for k in cutlass.range(1, k_iters, unroll=1):
                        # load k[k]
                        pipe_k.producer_acquire(state_k, pipe_k.producer_try_acquire(state_k))
                        shared.tma_copy(mK_g2s_atom, k_curr, sK, self.tile_k, self.dim, k, 0, pipe_k, state_k)
                        # load v[k]
                        pipe_v.producer_acquire(state_v, pipe_v.producer_try_acquire(state_v))
                        shared.tma_copy(mV_g2s_atom, v_curr, sV, self.dim, self.tile_k, 0, k, pipe_v, state_v)
                        state_k.advance()
                        state_v.advance()
    
    def _shared_cls(self, sQ_layout, sK_layout, sV_layout, sP_layout, sO_layout, sSum_layout):
        SharedStorage = type("SS", (), dict())
        items = [
            ('sQ_ptr', shared.memrange(self.dtype, sQ_layout, 1024)),
            ('sK_ptr', shared.memrange(self.dtype, sK_layout, 1024)),
            ('sV_ptr', shared.memrange(self.dtype, sV_layout, 1024)),
            ('sP_ptr', shared.memrange(self.dtype, sP_layout, 1024)),
            ('sO_ptr', shared.memrange(self.dtype, sO_layout, 1024)),
            ('sum_buf', shared.memrange(self.acc_dtype, sSum_layout, 1024)),
        ]
        for k, v in items:
            SharedStorage.__annotations__[k] = v

        BarrierStorage = type("BS1", (), dict())
        items = [
            ("k_pipe_ptr", cute.struct.MemRange[cutlass.Int64, self.stages * 2]),
            ("v_pipe_ptr", cute.struct.MemRange[cutlass.Int64, self.stages * 2]),
        ]
        for k, v in items:
            BarrierStorage.__annotations__[k] = v
        return cute.struct(SharedStorage), cute.struct(BarrierStorage)