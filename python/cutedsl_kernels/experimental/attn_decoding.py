import cutlass
from cutlass import cute
from cdsl_helpers import layout
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import scheduler
from cdsl_helpers import mma
from cdsl_helpers import reduction
from cdsl_helpers import elementwise
from cdsl_helpers import conversion
from cdsl_helpers import store


class Kernel:
  @cute.jit
  def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor):
    mQ = layout.select(mQ, (1, 2, 0))
    mK = layout.select(mK, (1, 2, 0))
    mV = layout.select(mV, (1, 2, 0))
    mO = layout.select(mO, (1, 2, 0))
    sQ_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 64, 128, 1)
    sK_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 3)
    sV_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 3)
    sO_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 64, 64, 2)
    pv_acc_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 128, False)
    tiled_mma_53 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 16, False)
    tiled_mma_69 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 128, False)
    tiled_mma_100 = mma.get_tiled_mma(cutlass.BFloat16, True, False, cutlass.Float32, 64, 128, True)
    tiled_mma_140 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 64, 128, False)
    mO_tma_atom_1, mO_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(mO, sO_layout, 64, 64)
    mQ_tma_atom_2, mQ_tma_tensor_2 = shared.get_tma_tensor_and_atom(mQ, sQ_layout, 64, 128, 1)
    mK_tma_atom_3, mK_tma_tensor_3 = shared.get_tma_tensor_and_atom(mK, sK_layout, 128, 128, 1)
    mV_tma_atom_4, mV_tma_tensor_4 = shared.get_tma_tensor_and_atom(mV, sV_layout, 128, 128, 1)
    self.kernel(mQ, mK, mV, mO, sQ_layout, sK_layout, sV_layout, sO_layout, pv_acc_tiled_mma, tiled_mma_53, tiled_mma_69, tiled_mma_100, tiled_mma_140, mO_tma_atom_1, mO_tma_tensor_1, mQ_tma_atom_2, mQ_tma_tensor_2, mK_tma_atom_3, mK_tma_tensor_3, mV_tma_atom_4, mV_tma_tensor_4).launch(grid=[132, 1, 1], block=256)

  @cute.kernel
  def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor, sQ_layout, sK_layout, sV_layout, sO_layout, pv_acc_tiled_mma, tiled_mma_53, tiled_mma_69, tiled_mma_100, tiled_mma_140, mO_tma_atom_1, mO_tma_tensor_1, mQ_tma_atom_2, mQ_tma_tensor_2, mK_tma_atom_3, mK_tma_tensor_3, mV_tma_atom_4, mV_tma_tensor_4):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'sQ_ptr', cutlass.BFloat16, sQ_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sK_ptr', cutlass.BFloat16, sK_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sV_ptr', cutlass.BFloat16, sV_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sO_ptr', cutlass.BFloat16, sO_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'pipe_q_ptr', 1)
    shared.smem_add_barrier_array(SharedStorage_t, 'pipe_k_ptr', 3)
    shared.smem_add_barrier_array(SharedStorage_t, 'pipe_v_ptr', 3)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    sQ = shared.smem_get_tensor(smem_, 'sQ_ptr', sQ_layout)
    sK = shared.smem_get_tensor(smem_, 'sK_ptr', sK_layout)
    sV = shared.smem_get_tensor(smem_, 'sV_ptr', sV_layout)
    sO = shared.smem_get_tensor(smem_, 'sO_ptr', sO_layout)
    pipe_q = pipeline.make_tma_pipeline_alt(smem_, 'pipe_q_ptr', 1, shared.staged_tensor_sizes(cutlass.BFloat16, sQ_layout), 4, None, 1)
    pipe_k = pipeline.make_tma_pipeline_alt(smem_, 'pipe_k_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, sK_layout), 4, None, 1)
    pipe_v = pipeline.make_tma_pipeline_alt(smem_, 'pipe_v_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, sV_layout), 4, None, 1)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()

    # for q_len=64 kv_len=4096 dim=128 nheads=32

    if warpidx_ >= 0 and warpidx_ < 4:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      # Consumer
      state_c_q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
      state_c_k = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
      state_c_v = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 32, 132):
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((1, 32), (1, 1)), ((32, 1), (32, 32)), (32, 1), 1)
        pv_acc = mma.get_acc(pv_acc_tiled_mma, 64, 128, cutlass.Float32)
        pv_acc_accumulate = False
        sum_acc = reduction.make_mma_A_reduction_tensor(tiled_mma_53, 64, 16, cutlass.Float32)
        pipe_q.consumer_wait(state_c_q, pipe_q.consumer_try_wait(state_c_q))
        for k in cutlass.range(0, 32, 1):
          pipe_k.consumer_wait(state_c_k, pipe_k.consumer_try_wait(state_c_k))
          acc_69 = mma.single_gemm_ss(tidx_, 64, 128, tiled_mma_69, sQ, sK, state_c_q, state_c_k, 0)
          pipe_k.consumer_release(state_c_k)
          p_acc_scaled = cute.make_rmem_tensor_like(acc_69, cutlass.Float32)
          p_acc_scaled.store(elementwise.const_mul(acc_69.load(), 0.12751743082459868))
          p_exp = cute.make_rmem_tensor_like(p_acc_scaled, cutlass.Float32)
          p_exp.store(elementwise.exp2f(p_acc_scaled.load()))
          p_exp_frgA = cute.make_tensor(p_exp.iterator, layout.convert_layout_acc_frgA(p_exp.layout))
          reduction.row_sum_mixed_types(p_exp_frgA, sum_acc, cutlass.Float32)
          rt_240 = conversion.cvt_f16(p_exp_frgA, cutlass.BFloat16)
          pipe_v.consumer_wait(state_c_v, pipe_v.consumer_try_wait(state_c_v))
          mma.accumulating_gemm_rs(tidx_, tiled_mma_100, rt_240, layout.transpose_view(sV), pv_acc, state_c_v, pv_acc_accumulate, 0)
          pv_acc_accumulate = True
          pipe_v.consumer_release(state_c_v)
          state_c_k.advance()
          state_c_v.advance()
        pipe_q.consumer_release(state_c_q)
        state_c_q.advance()
        sum_acc.store(reduction.warp_sum_row_mma_layout(sum_acc.load()))
        rcp_rowsum = cute.make_rmem_tensor_like(sum_acc, cutlass.Float32)
        rcp_rowsum.store(elementwise.rcp(sum_acc.load()))
        acc_scaled = elementwise.row_mul(pv_acc, rcp_rowsum)
        o_slice = mO[None, None, sched_coord[0]]
        mO_tma_tensor_1_slice = mO_tma_tensor_1[None, None, sched_coord[0]]
        store.mma_epilogue_tma(tiled_mma_140, mO_tma_tensor_1_slice, mO_tma_atom_1, sO, acc_scaled, 64, 128, sched_coord[1], 0, tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 4 and warpidx_ < 8:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 4:
        warpidx_ = warpidx_ + 4
        tidx_ = tidx_ + 128
        # Producer
        state_p_q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
        state_p_k = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
        state_p_v = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 32, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((1, 32), (1, 1)), ((32, 1), (32, 32)), (32, 1), 1)
          q_slice = mQ[None, None, sched_coord[0]]
          k_slice = mK[None, None, sched_coord[0]]
          v_slice = mV[None, None, sched_coord[0]]
          pipe_q.producer_acquire(state_p_q, pipe_q.producer_try_acquire(state_p_q))
          mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info(None, -1)
          mQ_tma_tensor_2_slice = mQ_tma_tensor_2[None, None, sched_coord[0]]
          shared.tma_copy(mQ_tma_atom_2, mQ_tma_tensor_2_slice, sQ, 64, 128, sched_coord[1], 0, pipe_q, state_p_q, cta_coord_2, cta_layout_2, mcast_mask_2)
          for k in cutlass.range(0, 32, 1):
            pipe_k.producer_acquire(state_p_k, pipe_k.producer_try_acquire(state_p_k))
            mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info(None, -1)
            mK_tma_tensor_3_slice = mK_tma_tensor_3[None, None, sched_coord[0]]
            shared.tma_copy(mK_tma_atom_3, mK_tma_tensor_3_slice, sK, 128, 128, k, 0, pipe_k, state_p_k, cta_coord_3, cta_layout_3, mcast_mask_3)
            pipe_v.producer_acquire(state_p_v, pipe_v.producer_try_acquire(state_p_v))
            mcast_mask_4, cta_coord_4, cta_layout_4 = shared.get_multicast_info(None, -1)
            mV_tma_tensor_4_slice = mV_tma_tensor_4[None, None, sched_coord[0]]
            shared.tma_copy(mV_tma_atom_4, mV_tma_tensor_4_slice, sV, 128, 128, k, 0, pipe_v, state_p_v, cta_coord_4, cta_layout_4, mcast_mask_4)
            state_p_k.advance()
            state_p_v.advance()
          state_p_q.advance()