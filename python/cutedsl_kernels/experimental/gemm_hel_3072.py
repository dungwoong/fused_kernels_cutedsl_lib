import cutlass
from cutlass import cute
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import scheduler
from cdsl_helpers import mma
from cdsl_helpers import store
from cdsl_helpers import test

# kwargs={'tma_stages': 3}


class Kernel:
  @cute.jit
  def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    st_221_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 192, 64, 3)
    st_235_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 192, 64, 3)
    wgmma_acc_219_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 192, 64, 2)
    wgmma_acc_219_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 192, 192, False)
    tiled_mma_656 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 192, 16, True)
    tiled_mma_705 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 192, 192, True)
    tiled_mma_724 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 192, 192, False)
    c_tma_atom_1, c_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(c, wgmma_acc_219_epi_smem_layout, 192, 64)
    a_tma_atom_2, a_tma_tensor_2 = shared.get_tma_tensor_and_atom(a, st_221_layout, 192, 64, 1)
    b_tma_atom_3, b_tma_tensor_3 = shared.get_tma_tensor_and_atom(b, st_235_layout, 192, 64, 1)
    self.kernel(a, b, st_221_layout, st_235_layout, c, wgmma_acc_219_epi_smem_layout, wgmma_acc_219_tiled_mma, tiled_mma_656, tiled_mma_705, tiled_mma_724, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3).launch(grid=[132, 1, 1], block=512)

  @cute.kernel
  def kernel(self, a: cute.Tensor, b: cute.Tensor, st_221_layout, st_235_layout, c: cute.Tensor, wgmma_acc_219_epi_smem_layout, wgmma_acc_219_tiled_mma, tiled_mma_656, tiled_mma_705, tiled_mma_724, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_221_ptr', cutlass.BFloat16, st_221_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_235_ptr', cutlass.BFloat16, st_235_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_235_pipe_ptr', 3)
    shared.smem_add_shared_tensor(SharedStorage_t, 'wgmma_acc_219_epi_smem_ptr', cutlass.BFloat16, wgmma_acc_219_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    st_221 = shared.smem_get_tensor(smem_, 'st_221_ptr', st_221_layout)
    st_235 = shared.smem_get_tensor(smem_, 'st_235_ptr', st_235_layout)
    st_235_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_235_pipe_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, st_235_layout, st_221_layout), 12, None, 1)
    wgmma_acc_219_epi_smem = shared.smem_get_tensor(smem_, 'wgmma_acc_219_epi_smem_ptr', wgmma_acc_219_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    st_221_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_235_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_221_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_235_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    # test.print0('reached branching')
    if warpidx_ >= 0 and warpidx_ < 12:
      cute.arch.setmaxregister_increase(152)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 256, 132):
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (2, 2)), ((16, 1), (128, 8)), (16, 16), 8)
        wgmma_acc_219 = mma.get_acc(wgmma_acc_219_tiled_mma, 192, 192, cutlass.Float32)
        wgmma_acc_219_accumulate = False
        for k in cutlass.range(0, 48, 1):
          st_235_pipe.consumer_wait(st_221_cstate, st_235_pipe.consumer_try_wait(st_221_cstate))
        #   test.print0("waited")
          rt_231 = mma.copy_a_wgmma(tidx_, tiled_mma_656, st_221[None, None, st_221_cstate.index], 192, 64, cutlass.BFloat16)
          mma.accumulating_gemm_rs(tidx_, tiled_mma_705, rt_231, st_235, wgmma_acc_219, st_221_cstate, wgmma_acc_219_accumulate, -1)
          wgmma_acc_219_accumulate = True
          cute.nvgpu.warpgroup.wait_group(0)
          st_235_pipe.consumer_release(st_221_cstate)
        #   test.print0("released")
          st_221_cstate.advance()
          st_235_pstate.advance()
        # test.print0("reached epi")
        store.mma_epilogue_tma(tiled_mma_724, c_tma_tensor_1, c_tma_atom_1, wgmma_acc_219_epi_smem, wgmma_acc_219, 192, 192, sched_coord[0], sched_coord[1], tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 12 and warpidx_ < 16:
      cute.arch.setmaxregister_decrease(56)
      if warpidx_ == 12:
        warpidx_ = warpidx_ + 12
        tidx_ = tidx_ + 384
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 256, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (2, 2)), ((16, 1), (128, 8)), (16, 16), 8)
          for k in cutlass.range(0, 48, 1):
            if cutlass.const_expr(True):
              st_235_pipe.producer_acquire(st_235_pstate, st_235_pipe.producer_try_acquire(st_235_pstate))
              mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info(None, -1)
              shared.tma_copy(a_tma_atom_2, a_tma_tensor_2, st_221, 192, 64, sched_coord[0], k, st_235_pipe, st_235_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info(None, -1)
              shared.tma_copy(b_tma_atom_3, b_tma_tensor_3, st_235, 192, 64, sched_coord[1], k, st_235_pipe, st_235_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
            st_235_pstate.advance()
