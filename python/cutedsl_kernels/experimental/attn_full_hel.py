import cutlass
from cutlass import cute
from cdsl_helpers import layout
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import reduction
from cdsl_helpers import mma
from cdsl_helpers import scheduler
from cdsl_helpers import elementwise
from cdsl_helpers import conversion
from cdsl_helpers import store

# kwargs={'tma_stages': 2}


class Kernel:
  @cute.jit
  def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor):
    mQ = layout.select(mQ, (1, 2, 0))
    mK = layout.select(mK, (1, 2, 0))
    mV = layout.select(mV, (1, 2, 0))
    st_5534_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 2)
    st_5552_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 2)
    st_5637_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 128, 2)
    mO = layout.select(mO, (1, 2, 0))
    rt_5657_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    tiled_mma_5524 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, False)
    tiled_mma_5590 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, False)
    wgmma_acc_5587_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    tiled_mma_6327 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    tiled_mma_6374 = mma.get_tiled_mma(cutlass.BFloat16, True, False, cutlass.Float32, 128, 128, True)
    tiled_mma_6392 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    mO_tma_atom_1, mO_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(mO, rt_5657_epi_smem_layout, 128, 64)
    mQ_tma_atom_2, mQ_tma_tensor_2 = shared.get_tma_tensor_and_atom(mQ, st_5534_layout, 128, 128, 1)
    mK_tma_atom_3, mK_tma_tensor_3 = shared.get_tma_tensor_and_atom(mK, st_5552_layout, 128, 128, 1)
    mV_tma_atom_4, mV_tma_tensor_4 = shared.get_tma_tensor_and_atom(mV, st_5637_layout, 128, 128, 1)
    self.kernel(mQ, mK, mV, st_5534_layout, st_5552_layout, st_5637_layout, mO, rt_5657_epi_smem_layout, tiled_mma_5524, tiled_mma_5590, wgmma_acc_5587_tiled_mma, tiled_mma_6327, tiled_mma_6374, tiled_mma_6392, mO_tma_atom_1, mO_tma_tensor_1, mQ_tma_atom_2, mQ_tma_tensor_2, mK_tma_atom_3, mK_tma_tensor_3, mV_tma_atom_4, mV_tma_tensor_4).launch(grid=[132, 1, 1], block=384, cluster=[1, 1, 1])

  @cute.kernel
  def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, st_5534_layout, st_5552_layout, st_5637_layout, mO: cute.Tensor, rt_5657_epi_smem_layout, tiled_mma_5524, tiled_mma_5590, wgmma_acc_5587_tiled_mma, tiled_mma_6327, tiled_mma_6374, tiled_mma_6392, mO_tma_atom_1, mO_tma_tensor_1, mQ_tma_atom_2, mQ_tma_tensor_2, mK_tma_atom_3, mK_tma_tensor_3, mV_tma_atom_4, mV_tma_tensor_4):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_5534_ptr', cutlass.BFloat16, st_5534_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_5534_pipe_ptr', 2)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_5552_ptr', cutlass.BFloat16, st_5552_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_5552_pipe_ptr', 2)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_5637_ptr', cutlass.BFloat16, st_5637_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_5637_pipe_ptr', 2)
    shared.smem_add_shared_tensor(SharedStorage_t, 'rt_5657_epi_smem_ptr', cutlass.BFloat16, rt_5657_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    st_5534 = shared.smem_get_tensor(smem_, 'st_5534_ptr', st_5534_layout)
    st_5534_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_5534_pipe_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, st_5534_layout), 8, cute.make_layout((1, 1, 1, 1)), 1)
    st_5552 = shared.smem_get_tensor(smem_, 'st_5552_ptr', st_5552_layout)
    st_5552_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_5552_pipe_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, st_5552_layout), 8, cute.make_layout((1, 1, 1, 1)), 1)
    st_5637 = shared.smem_get_tensor(smem_, 'st_5637_ptr', st_5637_layout)
    st_5637_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_5637_pipe_ptr', 2, shared.staged_tensor_sizes(cutlass.BFloat16, st_5637_layout), 8, cute.make_layout((1, 1, 1, 1)), 1)
    rt_5657_epi_smem = shared.smem_get_tensor(smem_, 'rt_5657_epi_smem_ptr', rt_5657_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    st_5534_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
    st_5552_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
    st_5637_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
    st_5534_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 2)
    st_5552_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 2)
    st_5637_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 2)
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 1024, 132):
        rt_5523 = reduction.make_mma_A_ninf_tensor(tiled_mma_5524, 128, 16, cutlass.Float32)
        rt_5589 = reduction.make_mma_A_reduction_tensor(tiled_mma_5590, 128, 16, cutlass.Float32)
        wgmma_acc_5587 = mma.get_acc(wgmma_acc_5587_tiled_mma, 128, 128, cutlass.Float32)
        wgmma_acc_5587_accumulate = False
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((32, 32), (1, 1)), ((32, 1), (1024, 32)), (32, 32), 32)
        gt_slice_5544 = mK[None, None, sched_coord[0]]
        gt_slice_5629 = mV[None, None, sched_coord[0]]
        gt_slice_5526 = mQ[None, None, sched_coord[0]]
        gt_slice_5663 = mO[None, None, sched_coord[0]]
        st_5534_pipe.consumer_wait(st_5534_cstate, st_5534_pipe.consumer_try_wait(st_5534_cstate))
        for k in cutlass.range(0, 32, 1):
          rt_5569 = cute.make_rmem_tensor_like(rt_5523, cutlass.Float32)
          rt_5569.store(elementwise.copy_elementwise(rt_5523.load()))
          st_5552_pipe.consumer_wait(st_5552_cstate, st_5552_pipe.consumer_try_wait(st_5552_cstate))
          acc_6327 = mma.single_gemm_ss(tidx_, 128, 128, tiled_mma_6327, st_5534, st_5552, st_5534_cstate, st_5552_cstate, 0)
          st_5552_pipe.consumer_release(st_5552_cstate)
          rt_5565 = cute.make_rmem_tensor_like(acc_6327, cutlass.Float32)
          rt_5565.store(elementwise.const_mul(acc_6327.load(), 0.12751743082459868))
          reduction.row_max_f32(rt_5565, rt_5523)
          rt_5523.store(reduction.warp_max_row_mma_layout(rt_5523.load()))
          rt_5581 = elementwise.row_bcast_sub(rt_5565, rt_5523)
          rt_5592 = cute.make_rmem_tensor_like(rt_5581, cutlass.Float32)
          rt_5592.store(elementwise.exp2f(rt_5581.load()))
          rt_5592_frgA = cute.make_tensor(rt_5592.iterator, layout.convert_layout_acc_frgA(rt_5592.layout))
          rt_5602 = elementwise.tilewise_sub(rt_5569, rt_5523)
          rt_5608 = cute.make_rmem_tensor_like(rt_5602, cutlass.Float32)
          rt_5608.store(elementwise.exp2f(rt_5602.load()))
          rt_5589.store(elementwise.tilewise_mul(rt_5589, rt_5608).load())
          wgmma_acc_5587.store(elementwise.row_mul(wgmma_acc_5587, rt_5608).load())
          reduction.row_sum_mixed_types(rt_5592_frgA, rt_5589, cutlass.Float32)
          rt_6888 = conversion.cvt_f16(rt_5592_frgA, cutlass.BFloat16)
          st_5552_pstate.advance()
          st_5552_cstate.advance()
          st_5637_pipe.consumer_wait(st_5637_cstate, st_5637_pipe.consumer_try_wait(st_5637_cstate))
          mma.accumulating_gemm_rs(tidx_, tiled_mma_6374, rt_6888, layout.transpose_view(st_5637), wgmma_acc_5587, st_5637_cstate, wgmma_acc_5587_accumulate, -1)
          wgmma_acc_5587_accumulate = True
          cute.nvgpu.warpgroup.wait_group(0)
          st_5637_pipe.consumer_release(st_5637_cstate)
          st_5637_pstate.advance()
          st_5637_cstate.advance()
        st_5534_pipe.consumer_release(st_5534_cstate)
        rt_5589.store(reduction.warp_sum_row_mma_layout(rt_5589.load()))
        rt_5625 = cute.make_rmem_tensor_like(rt_5589, cutlass.Float32)
        rt_5625.store(elementwise.rcp(rt_5589.load()))
        rt_5657 = elementwise.row_mul(wgmma_acc_5587, rt_5625)
        st_5534_pstate.advance()
        st_5534_cstate.advance()
        mO_tma_tensor_1_slice = mO_tma_tensor_1[None, None, sched_coord[0]]
        store.mma_epilogue_tma(tiled_mma_6392, mO_tma_tensor_1_slice, mO_tma_atom_1, rt_5657_epi_smem, rt_5657, 128, 128, sched_coord[1], 0, tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 1024, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((32, 32), (1, 1)), ((32, 1), (1024, 32)), (32, 32), 32)
          gt_slice_5544 = mK[None, None, sched_coord[0]]
          gt_slice_5629 = mV[None, None, sched_coord[0]]
          gt_slice_5526 = mQ[None, None, sched_coord[0]]
          if cutlass.const_expr(True):
            st_5534_pipe.producer_acquire(st_5534_pstate, st_5534_pipe.producer_try_acquire(st_5534_pstate))
            mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info([1, 1, 1], -1)
            mQ_tma_tensor_2_slice = mQ_tma_tensor_2[None, None, sched_coord[0]]
            shared.tma_copy(mQ_tma_atom_2, mQ_tma_tensor_2_slice, st_5534, 128, 128, sched_coord[1], 0, st_5534_pipe, st_5534_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
          for k in cutlass.range(0, 32, 1):
            if cutlass.const_expr(True):
              st_5552_pipe.producer_acquire(st_5552_pstate, st_5552_pipe.producer_try_acquire(st_5552_pstate))
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info([1, 1, 1], -1)
              mK_tma_tensor_3_slice = mK_tma_tensor_3[None, None, sched_coord[0]]
              shared.tma_copy(mK_tma_atom_3, mK_tma_tensor_3_slice, st_5552, 128, 128, k, 0, st_5552_pipe, st_5552_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
            st_5552_pstate.advance()
            if cutlass.const_expr(True):
              st_5637_pipe.producer_acquire(st_5637_pstate, st_5637_pipe.producer_try_acquire(st_5637_pstate))
              mcast_mask_4, cta_coord_4, cta_layout_4 = shared.get_multicast_info([1, 1, 1], -1)
              mV_tma_tensor_4_slice = mV_tma_tensor_4[None, None, sched_coord[0]]
              shared.tma_copy(mV_tma_atom_4, mV_tma_tensor_4_slice, st_5637, 128, 128, k, 0, st_5637_pipe, st_5637_pstate, cta_coord_4, cta_layout_4, mcast_mask_4)
            st_5637_pstate.advance()
          st_5534_pstate.advance()