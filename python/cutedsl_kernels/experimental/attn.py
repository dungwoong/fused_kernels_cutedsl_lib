import cutlass
from cutlass import cute
from cdsl_helpers import layout
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import scheduler
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


class Kernel:
  @cute.jit
  def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor):
    mQ = layout.select(mQ, (1, 2, 0))
    mK = layout.select(mK, (1, 2, 0))
    mV = layout.select(mV, (1, 2, 0))
    mO = layout.select(mO, (1, 2, 0))
    sQ_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 1)
    sK_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 2)
    sV_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 2)
    sO_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    mQ_tma_atom_1, mQ_tma_tensor_1 = shared.get_tma_tensor_and_atom(mQ, sQ_layout, 128, 128, 1)
    self.kernel(mQ, mK, mV, mO, sQ_layout, sK_layout, sV_layout, sO_layout, mQ_tma_atom_1, mQ_tma_tensor_1).launch(grid=[132, 1, 1], block=384)

  @cute.kernel
  def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor, sQ_layout, sK_layout, sV_layout, sO_layout, mQ_tma_atom_1, mQ_tma_tensor_1):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'sQ_ptr', cutlass.BFloat16, sQ_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sK_ptr', cutlass.BFloat16, sK_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sV_ptr', cutlass.BFloat16, sV_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sO_ptr', cutlass.BFloat16, sO_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'pipe_q_ptr', 1)
    shared.smem_add_barrier_array(SharedStorage_t, 'pipe_k_ptr', 2)
    shared.smem_add_barrier_array(SharedStorage_t, 'pipe_v_ptr', 2)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    sQ = shared.smem_get_tensor(smem_, 'sQ_ptr', sQ_layout)
    sK = shared.smem_get_tensor(smem_, 'sK_ptr', sK_layout)
    sV = shared.smem_get_tensor(smem_, 'sV_ptr', sV_layout)
    sO = shared.smem_get_tensor(smem_, 'sO_ptr', sO_layout)
    # pipe_k = pipeline.make_tma_pipeline_alt(smem_, 'pipe_k_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, sK_layout), 8, None, 1)
    # pipe_v = pipeline.make_tma_pipeline_alt(smem_, 'pipe_v_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, sV_layout), 8, None, 1)
    pipe_q = pipeline.make_tma_pipeline_alt(smem_, 'pipe_q_ptr', 1, shared.staged_tensor_sizes(cutlass.BFloat16, sQ_layout), 8, None, 1)
    pipe_k = pipeline.make_tma_pipeline_alt(smem_, 'pipe_k_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, sK_layout), 8, None, 1)
    pipe_v = pipeline.make_tma_pipeline_alt(smem_, 'pipe_v_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, sV_layout), 8, None, 1)

    print('pipe_q', pipe_q)
    print('pipe_k', pipe_k)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()

    # for q_len=4096 kv_len=4096 dim=128 nheads=32

    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      # Consumer
      state_c_q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 1024, 132):
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((32, 32), (1, 1)), ((32, 1), (1024, 32)), (32, 32), 32)
        pipe_q.consumer_wait(state_c_q, pipe_q.consumer_try_wait(state_c_q))
        pipe_q.consumer_release(state_c_q)
        state_c_q.advance()
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        # Producer
        state_p_q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 1024, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((32, 32), (1, 1)), ((32, 1), (1024, 32)), (32, 32), 32)
          q_slice = mQ[None, None, sched_coord[0]]
          k_slice = mK[None, None, sched_coord[0]]
          v_slice = mV[None, None, sched_coord[0]]
          pipe_q.producer_acquire(state_p_q, pipe_q.producer_try_acquire(state_p_q))
          mcast_mask_1, cta_coord_1, cta_layout_1 = shared.get_multicast_info(None, -1)
          mQ_tma_tensor_1_slice = mQ_tma_tensor_1[None, None, sched_coord[0]]
          shared.tma_copy(mQ_tma_atom_1, mQ_tma_tensor_1_slice, sQ, 128, 128, sched_coord[1], 0, pipe_q, state_p_q, cta_coord_1, cta_layout_1, mcast_mask_1)
          state_p_q.advance()