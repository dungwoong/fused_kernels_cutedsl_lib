import cutlass
from cutlass import cute
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import scheduler
from cdsl_helpers import mma
from cdsl_helpers import store

# kwargs={'tma_stages': 3}


class Kernel:
  @cute.jit
  def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    st_571_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    st_581_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 256, 64, 3)
    wgmma_acc_569_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    wgmma_acc_569_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_699 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_720 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    c_tma_atom_1, c_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(c, wgmma_acc_569_epi_smem_layout, 128, 64)
    a_tma_atom_2, a_tma_tensor_2 = shared.get_tma_tensor_and_atom(a, st_571_layout, 128, 64, 1)
    b_tma_atom_3, b_tma_tensor_3 = shared.get_tma_tensor_and_atom(b, st_581_layout, 256, 64, 1)
    self.kernel(st_571_layout, st_581_layout, a, b, c, wgmma_acc_569_epi_smem_layout, wgmma_acc_569_tiled_mma, tiled_mma_699, tiled_mma_720, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3).launch(grid=[132, 1, 1], block=384)

  @cute.kernel
  def kernel(self, st_571_layout, st_581_layout, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, wgmma_acc_569_epi_smem_layout, wgmma_acc_569_tiled_mma, tiled_mma_699, tiled_mma_720, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_571_ptr', cutlass.BFloat16, st_571_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_581_ptr', cutlass.BFloat16, st_581_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_581_pipe_ptr', 3)
    shared.smem_add_shared_tensor(SharedStorage_t, 'wgmma_acc_569_epi_smem_ptr', cutlass.BFloat16, wgmma_acc_569_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    st_571 = shared.smem_get_tensor(smem_, 'st_571_ptr', st_571_layout)
    st_581 = shared.smem_get_tensor(smem_, 'st_581_ptr', st_581_layout)
    st_581_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_581_pipe_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, st_581_layout, st_571_layout), 8, None, 1)
    wgmma_acc_569_epi_smem = shared.smem_get_tensor(smem_, 'wgmma_acc_569_epi_smem_ptr', wgmma_acc_569_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    # st_571_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_571_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_581_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    # st_581_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 512, 132):
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (2, 4)), ((32, 1), (256, 8)), (32, 16), 8)
        wgmma_acc_569 = mma.get_acc(wgmma_acc_569_tiled_mma, 128, 256, cutlass.Float32)
        wgmma_acc_569_accumulate = False
        for k in cutlass.range(0, 64, 1):
          st_581_pipe.consumer_wait(st_571_cstate, st_581_pipe.consumer_try_wait(st_571_cstate))
          mma.accumulating_gemm_ss(tidx_, tiled_mma_699, st_571, st_581, wgmma_acc_569, st_571_cstate, st_571_cstate, wgmma_acc_569_accumulate, -1)
          wgmma_acc_569_accumulate = True
          cute.nvgpu.warpgroup.wait_group(0)
          st_581_pipe.consumer_release(st_571_cstate)
          st_571_cstate.advance()
        #   st_581_pstate.advance()
        store.mma_epilogue_tma(tiled_mma_720, c_tma_tensor_1, c_tma_atom_1, wgmma_acc_569_epi_smem, wgmma_acc_569, 128, 256, sched_coord[0], sched_coord[1], tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 512, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (2, 4)), ((32, 1), (256, 8)), (32, 16), 8)
          for k in cutlass.range(0, 64, 1):
            if cutlass.const_expr(True):
              st_581_pipe.producer_acquire(st_581_pstate, st_581_pipe.producer_try_acquire(st_581_pstate))
              mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info(None, -1)
              shared.tma_copy(a_tma_atom_2, a_tma_tensor_2, st_571, 128, 64, sched_coord[0], k, st_581_pipe, st_581_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info(None, -1)
              shared.tma_copy(b_tma_atom_3, b_tma_tensor_3, st_581, 256, 64, sched_coord[1], k, st_581_pipe, st_581_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
            st_581_pstate.advance()