import cutlass
from cutlass import cute
from cdsl_helpers import layout
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import reduction
from cdsl_helpers import scheduler
from cdsl_helpers import mma
from cdsl_helpers import elementwise
from cdsl_helpers import conversion
from cdsl_helpers import store, test

# kwargs={'tma_stages': 2}


class Kernel:
  @cute.jit
  def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor):
    mQ = layout.select(mQ, (1, 2, 0))
    mK = layout.select(mK, (1, 2, 0))
    mV = layout.select(mV, (1, 2, 0))
    sQ_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 1)
    sK_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 2)
    sV_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 2)
    mO = layout.select(mO, (1, 2, 0))
    acc_scaled_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    tiled_mma_44 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, False)
    tiled_mma_41 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, False)
    pv_acc_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    tiled_mma_279 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    tiled_mma_196 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, True)
    tiled_mma_287 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, True)
    tiled_mma_292 = mma.get_tiled_mma(cutlass.BFloat16, True, False, cutlass.Float32, 128, 128, True)
    tiled_mma_310 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    mO_tma_atom_1, mO_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(mO, acc_scaled_epi_smem_layout, 128, 64)
    mQ_tma_atom_2, mQ_tma_tensor_2 = shared.get_tma_tensor_and_atom(mQ, sQ_layout, 128, 128, 1)
    mK_tma_atom_3, mK_tma_tensor_3 = shared.get_tma_tensor_and_atom(mK, sK_layout, 128, 128, 1)
    mV_tma_atom_4, mV_tma_tensor_4 = shared.get_tma_tensor_and_atom(mV, sV_layout, 128, 128, 1)
    self.kernel(mQ, mK, mV, sQ_layout, sK_layout, sV_layout, mO, acc_scaled_epi_smem_layout, tiled_mma_44, tiled_mma_41, pv_acc_tiled_mma, tiled_mma_196, tiled_mma_287, tiled_mma_292, tiled_mma_310, mO_tma_atom_1, mO_tma_tensor_1, mQ_tma_atom_2, mQ_tma_tensor_2, mK_tma_atom_3, mK_tma_tensor_3, mV_tma_atom_4, mV_tma_tensor_4, tiled_mma_279).launch(grid=[132, 1, 1], block=384, cluster=[1, 1, 1])

  @cute.kernel
  def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, sQ_layout, sK_layout, sV_layout, mO: cute.Tensor, acc_scaled_epi_smem_layout, tiled_mma_44, tiled_mma_41, pv_acc_tiled_mma, tiled_mma_196, tiled_mma_287, tiled_mma_292, tiled_mma_310, mO_tma_atom_1, mO_tma_tensor_1, mQ_tma_atom_2, mQ_tma_tensor_2, mK_tma_atom_3, mK_tma_tensor_3, mV_tma_atom_4, mV_tma_tensor_4, tiled_mma_279):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'sQ_ptr', cutlass.BFloat16, sQ_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'sQ_pipe_ptr', 1)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sK_ptr', cutlass.BFloat16, sK_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'sK_pipe_ptr', 2)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sV_ptr', cutlass.BFloat16, sV_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'sV_pipe_ptr', 2)
    shared.smem_add_shared_tensor(SharedStorage_t, 'acc_scaled_epi_smem_ptr', cutlass.BFloat16, acc_scaled_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    sQ = shared.smem_get_tensor(smem_, 'sQ_ptr', sQ_layout)
    sQ_pipe = pipeline.make_tma_pipeline_alt(smem_, 'sQ_pipe_ptr', 1, shared.staged_tensor_sizes(cutlass.BFloat16, sQ_layout), 8, cute.make_layout((1, 1, 1, 1)), 1)
    sK = shared.smem_get_tensor(smem_, 'sK_ptr', sK_layout)
    sK_pipe = pipeline.make_tma_pipeline_alt(smem_, 'sK_pipe_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, sK_layout), 8, cute.make_layout((1, 1, 1, 1)), 1)
    sV = shared.smem_get_tensor(smem_, 'sV_ptr', sV_layout)
    sV_pipe = pipeline.make_tma_pipeline_alt(smem_, 'sV_pipe_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, sV_layout), 8, cute.make_layout((1, 1, 1, 1)), 1)
    acc_scaled_epi_smem = shared.smem_get_tensor(smem_, 'acc_scaled_epi_smem_ptr', acc_scaled_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    sQ_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
    sK_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
    sV_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
    sQ_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
    sK_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 2)
    sV_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 2)
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 1024, 132):
        max_acc = reduction.make_mma_A_ninf_tensor(tiled_mma_44, 128, 16, cutlass.Float32)
        sum_acc = reduction.make_mma_A_reduction_tensor(tiled_mma_41, 128, 16, cutlass.Float32)
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((32, 32), (1, 1)), ((32, 1), (1024, 32)), (32, 32), 32)
        pv_acc = mma.get_acc(pv_acc_tiled_mma, 128, 128, cutlass.Float32)
        pv_acc_accumulate = False
        k_slice = mK[None, None, sched_coord[0]]
        v_slice = mV[None, None, sched_coord[0]]
        q_slice = mQ[None, None, sched_coord[0]]
        o_slice = mO[None, None, sched_coord[0]]
        sQ_pipe.consumer_wait(sQ_cstate, sQ_pipe.consumer_try_wait(sQ_cstate))
        test.print0('hi')
        rQ = mma.copy_a_wgmma(tidx_, tiled_mma_196, sQ[None, None, sQ_cstate.index], 128, 128, cutlass.BFloat16)
        q_vals = rQ.load()

        # test.print0(rQ)
        for k in cutlass.range(0, 32, 1):
          max_acc_last = cute.make_rmem_tensor_like(max_acc, cutlass.Float32)
          max_acc_last.store(elementwise.copy_elementwise(max_acc.load()))
          sK_pipe.consumer_wait(sK_cstate, sK_pipe.consumer_try_wait(sK_cstate))
          
          rQ1 = cute.make_rmem_tensor_like(rQ)
          rQ1.store(q_vals)
          acc_287 = mma.single_gemm_rs(tidx_, 128, 128, tiled_mma_287, rQ1, sK, sK_cstate, 0)
          # acc_279 = mma.single_gemm_ss(tidx_, 128, 128, tiled_mma_279, sQ, sK, sQ_cstate, sK_cstate, 0)
          # test.print0(sQ[None, None, sQ_cstate.index])
          # test.print0(rQ)
          # test.print0(acc_279)
          test.print0(acc_287)
          # test.print0(k)
          sK_pipe.consumer_release(sK_cstate)
          p_acc_scaled = cute.make_rmem_tensor_like(acc_287, cutlass.Float32)
          p_acc_scaled.store(elementwise.const_mul(acc_287.load(), 0.12751743082459868))
          reduction.row_max_f32(p_acc_scaled, max_acc)
          max_acc.store(reduction.warp_max_row_mma_layout(max_acc.load()))
          p_scaled_sub = elementwise.row_bcast_sub(p_acc_scaled, max_acc)
          p_exp = cute.make_rmem_tensor_like(p_scaled_sub, cutlass.Float32)
          p_exp.store(elementwise.exp2f(p_scaled_sub.load()))
          p_exp_frgA = cute.make_tensor(p_exp.iterator, layout.convert_layout_acc_frgA(p_exp.layout))
          max_sub = elementwise.tilewise_sub(max_acc_last, max_acc)
          exp_diff = cute.make_rmem_tensor_like(max_sub, cutlass.Float32)
          exp_diff.store(elementwise.exp2f(max_sub.load()))
          sum_acc.store(elementwise.tilewise_mul(sum_acc, exp_diff).load())
          pv_acc.store(elementwise.row_mul(pv_acc, exp_diff).load())
          reduction.row_sum_mixed_types(p_exp_frgA, sum_acc, cutlass.Float32)
          rt_814 = conversion.cvt_f16(p_exp_frgA, cutlass.BFloat16)
          sK_pstate.advance()
          sK_cstate.advance()
          sV_pipe.consumer_wait(sV_cstate, sV_pipe.consumer_try_wait(sV_cstate))
          mma.accumulating_gemm_rs(tidx_, tiled_mma_292, rt_814, layout.transpose_view(sV), pv_acc, sV_cstate, pv_acc_accumulate, -1)
          pv_acc_accumulate = True
          cute.nvgpu.warpgroup.wait_group(0)
          sV_pipe.consumer_release(sV_cstate)
          sV_pstate.advance()
          sV_cstate.advance()
        sum_acc.store(reduction.warp_sum_row_mma_layout(sum_acc.load()))
        rcp_rowsum = cute.make_rmem_tensor_like(sum_acc, cutlass.Float32)
        rcp_rowsum.store(elementwise.rcp(sum_acc.load()))
        acc_scaled = elementwise.row_mul(pv_acc, rcp_rowsum)
        sQ_pipe.consumer_release(sQ_cstate)
        sQ_pstate.advance()
        sQ_cstate.advance()
        mO_tma_tensor_1_slice = mO_tma_tensor_1[None, None, sched_coord[0]]
        store.mma_epilogue_tma(tiled_mma_310, mO_tma_tensor_1_slice, mO_tma_atom_1, acc_scaled_epi_smem, acc_scaled, 128, 128, sched_coord[1], 0, tidx_, warpidx_, cutlass.Float32)

      # for q_len=4096 kv_len=4096 dim=128 nheads=32

    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 1024, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((32, 32), (1, 1)), ((32, 1), (1024, 32)), (32, 32), 32)
          k_slice = mK[None, None, sched_coord[0]]
          v_slice = mV[None, None, sched_coord[0]]
          q_slice = mQ[None, None, sched_coord[0]]
          if cutlass.const_expr(True):
            sQ_pipe.producer_acquire(sQ_pstate, sQ_pipe.producer_try_acquire(sQ_pstate))
            mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info([1, 1, 1], -1)
            mQ_tma_tensor_2_slice = mQ_tma_tensor_2[None, None, sched_coord[0]]
            shared.tma_copy(mQ_tma_atom_2, mQ_tma_tensor_2_slice, sQ, 128, 128, sched_coord[1], 0, sQ_pipe, sQ_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
          for k in cutlass.range(0, 32, 1):
            if cutlass.const_expr(True):
              sK_pipe.producer_acquire(sK_pstate, sK_pipe.producer_try_acquire(sK_pstate))
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info([1, 1, 1], -1)
              mK_tma_tensor_3_slice = mK_tma_tensor_3[None, None, sched_coord[0]]
              shared.tma_copy(mK_tma_atom_3, mK_tma_tensor_3_slice, sK, 128, 128, k, 0, sK_pipe, sK_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
            sK_pstate.advance()
            if cutlass.const_expr(True):
              sV_pipe.producer_acquire(sV_pstate, sV_pipe.producer_try_acquire(sV_pstate))
              mcast_mask_4, cta_coord_4, cta_layout_4 = shared.get_multicast_info([1, 1, 1], -1)
              mV_tma_tensor_4_slice = mV_tma_tensor_4[None, None, sched_coord[0]]
              shared.tma_copy(mV_tma_atom_4, mV_tma_tensor_4_slice, sV, 128, 128, k, 0, sV_pipe, sV_pstate, cta_coord_4, cta_layout_4, mcast_mask_4)
            sV_pstate.advance()
          sQ_pstate.advance()