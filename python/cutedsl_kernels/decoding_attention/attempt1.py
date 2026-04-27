from typing import Tuple

import cutlass
from cutlass import cute, pipeline
from cdsl_helpers import shared, mma, pipeline as my_pipeline
from . import attn_scheduler


class Kernel:
    """
    Sample shapes:
    X: (16, 4096, nheads)
    Wqkv: (128, 4096, nheads)
    
    """

    def __init__(
            self,
            qkw_mnk: Tuple[int, int, int],
            stg1_stages: int,
            persistent: bool,
            ):
        self.dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.m1, self.n1, self.k1 = qkw_mnk # stage 1
        self.stg1_stages = stg1_stages

        self.nconsumer_warps = None

        self.consumer_regs = 232
        self.producer_regs = 40

        self.is_persistent = persistent
    
    # m means matrix e.g. mX
    # s means shared tensor e.g. sX

    @cute.jit
    def __call__(self, mX: cute.Tensor, mWq: cute.Tensor, mWk: cute.Tensor, mWv: cute.Tensor, mKcache: cute.Tensor, mVcache: cute.Tensor):
        """
        e.g. 
        32 heads, dim=128 --> model dim = 4096
        mX = (16, 4096)
        mWq, k, v = (4096, 4096) (need to multiply by columns corresponding to heads)

        mKcache, mVcache = (32, 1024, 128)
        """
        print(mKcache)
        sX_layout = shared.get_smem_layout_row_major(self.dtype, self.m1, self.k1, self.stg1_stages)
        sWq_layout = shared.get_smem_layout_row_major(self.dtype, self.n1, self.k1, self.stg1_stages)
        sWk_layout = shared.get_smem_layout_row_major(self.dtype, self.n1, self.k1, self.stg1_stages)
        sWv_layout = shared.get_smem_layout_row_major(self.dtype, self.n1, self.k1, self.stg1_stages)

        # Swap M and N. Tiled gemm is reused across all matmuls for stage 1
        tiled_gemm_1 = mma.get_tiled_mma(self.dtype, True, True, self.acc_dtype, self.n1, self.m1)

        consumer_wgs = tiled_gemm_1.size // 128
        self.nconsumer_warps = consumer_wgs * 4

        mX_g2s_atom, mX_g2s_tensor = shared.get_tma_tensor_and_atom(mX, sX_layout, self.m1, self.k1)
        mWq_g2s_atom, mWq_g2s_tensor = shared.get_tma_tensor_and_atom(mWq, sWq_layout, self.n1, self.k1)
        mWk_g2s_atom, mWk_g2s_tensor = shared.get_tma_tensor_and_atom(mWk, sWk_layout, self.n1, self.k1)
        mWv_g2s_atom, mWv_g2s_tensor = shared.get_tma_tensor_and_atom(mWv, sWv_layout, self.n1, self.k1)

        # normally we pass in output matrix but mX works here
        nheads = mKcache.shape[0]
        scheduler_params = attn_scheduler.HeadAttnTileScheduler.to_underlying_arguments(
            attn_scheduler.HeadAttnTileSchedulerArguments.create(nheads, self.is_persistent)
        )
        grid = attn_scheduler.HeadAttnTileScheduler.get_grid_shape(scheduler_params, 132)
        self.kernel(
            scheduler_params, sX_layout, sWq_layout, sWk_layout, sWv_layout, 
            mX_g2s_atom, mX_g2s_tensor, 
            mWq_g2s_atom, mWq_g2s_tensor, 
            mWk_g2s_atom, mWk_g2s_tensor, 
            mWv_g2s_atom, mWv_g2s_tensor,
            tiled_gemm_1).launch(grid=grid, block=[(self.nconsumer_warps + 4) * cute.arch.WARP_SIZE])

    @cute.kernel
    def kernel(
        self,
        scheduler_params,
        sX_layout, sWq_layout, sWk_layout, sWv_layout,
        mX_atom, mX,
        mWq_atom, mWq,
        mWk_atom, mWk,
        mWv_atom, mWv,
        tiled_gemm_1,
        ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # SMEM
        ss1_t, bars1_t = self._shared_stg1(sX_layout, sWq_layout, sWk_layout, sWv_layout)
        s_alloc = cutlass.utils.SmemAllocator()
        dsmem = s_alloc.allocate(ss1_t.size_in_bytes(), byte_alignment=1024)
        smem_bars1 = s_alloc.allocate(bars1_t)

        # S1 tensors
        s1_smem = ss1_t(dsmem)
        sX = shared.smem_get_tensor(s1_smem, 'sX_ptr', sX_layout)
        sWq = shared.smem_get_tensor(s1_smem, 'sWq_ptr', sWq_layout)
        sWk = shared.smem_get_tensor(s1_smem, 'sWk_ptr', sWk_layout)
        sWv = shared.smem_get_tensor(s1_smem, 'sWv_ptr', sWv_layout)
        
        x_bytes = cute.size_in_bytes(self.dtype, cute.select(sX_layout, mode=[0, 1]))
        q_bytes = cute.size_in_bytes(self.dtype, cute.select(sWq_layout, mode=[0, 1]))
        k_bytes = cute.size_in_bytes(self.dtype, cute.select(sWk_layout, mode=[0, 1]))
        v_bytes = cute.size_in_bytes(self.dtype, cute.select(sWv_layout, mode=[0, 1]))
        stg1_pipe = my_pipeline.make_tma_pipeline(
            smem_bars1.pipe_ptr.data_ptr(),
            self.stg1_stages,
            num_consumer_warps=self.nconsumer_warps,
            num_bytes=x_bytes + q_bytes + k_bytes + v_bytes,
            mcast_size=1,
            cta_layout_vmnk=None
        )

        scheduler = attn_scheduler.HeadAttnTileScheduler.create(scheduler_params)

        k_iters = cute.size(mX, mode=[1]) // self.k1
        if (warp_idx < self.nconsumer_warps): # Consumer
            cute.arch.setmaxregister_increase(self.consumer_regs)
            s1_cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stg1_stages)
            work_tile = scheduler.initial_work_tile_info()
            acc_q = mma.get_acc(tiled_gemm_1, self.n1, self.m1, self.acc_dtype)
            acc_k = mma.get_acc(tiled_gemm_1, self.n1, self.m1, self.acc_dtype)
            acc_v = mma.get_acc(tiled_gemm_1, self.n1, self.m1, self.acc_dtype)
            while work_tile.is_valid_tile:
                tile_coord = work_tile.tile_idx
                s1_cstate, tiled_gemm_1 = self.consumer_stg1(
                    stg1_pipe, s1_cstate, k_iters, tiled_gemm_1, tidx, sX, sWq, sWk, sWv, acc_q, acc_k, acc_v
                )
        if (warp_idx >= self.nconsumer_warps): # Producer
            cute.arch.setmaxregister_decrease(self.producer_regs)
            s1_pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stg1_stages)
            work_tile = scheduler.initial_work_tile_info()
            
            while work_tile.is_valid_tile:
                tile_coord = work_tile.tile_idx
                head_idx = tile_coord[0]
                s1_pstate = self.producer_stg1(
                    stg1_pipe, s1_pstate, k_iters, head_idx,
                    mX_atom, mX, sX, 
                    mWq_atom, mWq, sWq, 
                    mWk_atom, mWk, sWk, 
                    mWv_atom, mWv, sWv)
            
            stg1_pipe.producer_tail(s1_pstate)
    
    @cute.jit
    def consumer_stg1(
        self, pipe: pipeline.PipelineAsync, state: pipeline.PipelineState, 
        k_iters: cutlass.Int32, tiled_gemm: cute.TiledMma, tidx: cutlass.Int32,
        sX: cute.Tensor, sWq: cute.Tensor, sWk: cute.Tensor, sWv: cute.Tensor,
        acc_q: cute.Tensor, acc_k: cute.Tensor, acc_v: cute.Tensor):
        accumulate_stg1 = False
        for k in cutlass.range(k_iters, unroll=1):
            pipe.consumer_wait(state, pipe.consumer_try_wait(state))

            # Here, you get Q, K and V transposed by taking e.g. Wq^T @ X^T = Q^T
            mma.accumulating_gemm_ss(tidx, tiled_gemm, sWq, sX, acc_q, state, state, accumulate_stg1, -1)
            mma.accumulating_gemm_ss(tidx, tiled_gemm, sWk, sX, acc_k, state, state, accumulate_stg1, -1)
            mma.accumulating_gemm_ss(tidx, tiled_gemm, sWv, sX, acc_v, state, state, accumulate_stg1, -1)
            cute.nvgpu.warpgroup.wait_group(0)
            pipe.consumer_release(state)
            state.advance()
        return state, tiled_gemm

    @cute.jit
    def producer_stg1(self, pipe: pipeline.PipelineAsync, state: pipeline.PipelineState, k_iters: cutlass.Int32, head_idx: cutlass.Int32, tmaa_X, mX, sX, tmaa_Wq, mWq, sWq, tmaa_Wk, mWk, sWk, tmaa_Wv, mWv, sWv):
        """
        X, wQKV should be sliced to remove head dim
        """
        for k in cutlass.range(k_iters, unroll=1):
            pipe.producer_acquire(state, pipe.producer_try_acquire(state))
            shared.tma_copy(tmaa_X, mX, sX, self.m1, self.k1, 0, k, pipe, state)
            shared.tma_copy(tmaa_Wq, mWq, sWq, self.n1, self.k1, head_idx, k, pipe, state)
            shared.tma_copy(tmaa_Wk, mWk, sWk, self.n1, self.k1, head_idx, k, pipe, state)
            shared.tma_copy(tmaa_Wv, mWv, sWv, self.n1, self.k1, head_idx, k, pipe, state)
            state.advance()
        return state
    
    def _shared_stg1(self, sX_layout, sQ_layout, sK_layout, sV_layout):
        # Data storage
        SharedStorage = type("SS1", (), dict())
        items = [
            ('sX_ptr', shared.memrange(self.dtype, sX_layout, 1024)),
            ('sWq_ptr', shared.memrange(self.dtype, sQ_layout, 1024)),
            ('sWk_ptr', shared.memrange(self.dtype, sK_layout, 1024)),
            ('sWv_ptr', shared.memrange(self.dtype, sV_layout, 1024)),
            ]
        for k, v in items:
            SharedStorage.__annotations__[k] = v
        
        # Barrier storage
        # - I'll try using a single barrier, more barrier syncs isn't necessarily good
        BarrierStorage = type("BS1", (), dict())
        items = [
            ("pipe_ptr", cute.struct.MemRange[cutlass.Int64, self.stg1_stages * 2]),
        ]
        for k, v in items:
            BarrierStorage.__annotations__[k] = v

        return cute.struct(SharedStorage), cute.struct(BarrierStorage)