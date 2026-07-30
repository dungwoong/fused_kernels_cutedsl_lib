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
  def __call__(self, a: cute.Tensor, b: cute.Tensor, xa: cute.Tensor, lora_b: cute.Tensor, c: cute.Tensor):
    sA_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    sB_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 256, 64, 3)
    sXA_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 16, 1)
    sLB_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 256, 16, 1)
    acc_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    acc_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_162 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_261 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_282 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    c_tma_atom_1, c_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(c, acc_epi_smem_layout, 128, 64)
    xa_tma_atom_2, xa_tma_tensor_2 = shared.get_tma_tensor_and_atom(xa, sXA_layout, 128, 16, 1)
    lora_b_tma_atom_3, lora_b_tma_tensor_3 = shared.get_tma_tensor_and_atom(lora_b, sLB_layout, 256, 16, 1)
    a_tma_atom_4, a_tma_tensor_4 = shared.get_tma_tensor_and_atom(a, sA_layout, 128, 64, 1)
    b_tma_atom_5, b_tma_tensor_5 = shared.get_tma_tensor_and_atom(b, sB_layout, 256, 64, 1)
    self.kernel(a, b, xa, lora_b, sA_layout, sB_layout, sXA_layout, sLB_layout, c, acc_epi_smem_layout, acc_tiled_mma, tiled_mma_162, tiled_mma_261, tiled_mma_282, c_tma_atom_1, c_tma_tensor_1, xa_tma_atom_2, xa_tma_tensor_2, lora_b_tma_atom_3, lora_b_tma_tensor_3, a_tma_atom_4, a_tma_tensor_4, b_tma_atom_5, b_tma_tensor_5).launch(grid=[132, 1, 1], block=384)

  @cute.kernel
  def kernel(self, a: cute.Tensor, b: cute.Tensor, xa: cute.Tensor, lora_b: cute.Tensor, sA_layout, sB_layout, sXA_layout, sLB_layout, c: cute.Tensor, acc_epi_smem_layout, acc_tiled_mma, tiled_mma_162, tiled_mma_261, tiled_mma_282, c_tma_atom_1, c_tma_tensor_1, xa_tma_atom_2, xa_tma_tensor_2, lora_b_tma_atom_3, lora_b_tma_tensor_3, a_tma_atom_4, a_tma_tensor_4, b_tma_atom_5, b_tma_tensor_5):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'sA_ptr', cutlass.BFloat16, sA_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sB_ptr', cutlass.BFloat16, sB_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'sB_pipe_ptr', 3)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sXA_ptr', cutlass.BFloat16, sXA_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sLB_ptr', cutlass.BFloat16, sLB_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'sLB_pipe_ptr', 1)
    shared.smem_add_shared_tensor(SharedStorage_t, 'acc_epi_smem_ptr', cutlass.BFloat16, acc_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    sA = shared.smem_get_tensor(smem_, 'sA_ptr', sA_layout)
    sB = shared.smem_get_tensor(smem_, 'sB_ptr', sB_layout)
    sB_pipe = pipeline.make_tma_pipeline_alt(smem_, 'sB_pipe_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, sB_layout, sA_layout), 8, None, 1)
    sXA = shared.smem_get_tensor(smem_, 'sXA_ptr', sXA_layout)
    sLB = shared.smem_get_tensor(smem_, 'sLB_ptr', sLB_layout)
    sLB_pipe = pipeline.make_tma_pipeline_alt(smem_, 'sLB_pipe_ptr', 1, shared.staged_tensor_sizes(cutlass.BFloat16, sLB_layout, sXA_layout), 8, None, 1)
    acc_epi_smem = shared.smem_get_tensor(smem_, 'acc_epi_smem_ptr', acc_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    sA_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    sB_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    sXA_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
    sLB_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
    sA_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    sB_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    sXA_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
    sLB_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 512, 132):
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (2, 4)), ((32, 1), (256, 8)), (32, 16), 8)
        acc = mma.get_acc(acc_tiled_mma, 128, 256, cutlass.Float32)
        acc_accumulate = False
        for k in cutlass.range(0, 64, 1):
          sB_pipe.consumer_wait(sA_cstate, sB_pipe.consumer_try_wait(sA_cstate))
          mma.accumulating_gemm_ss(tidx_, tiled_mma_162, sA, sB, acc, sA_cstate, sA_cstate, acc_accumulate, -1)
          acc_accumulate = True
          cute.nvgpu.warpgroup.wait_group(0)
          sB_pipe.consumer_release(sA_cstate)
          sA_cstate.advance()
          sB_pstate.advance()
        sLB_pipe.consumer_wait(sXA_cstate, sLB_pipe.consumer_try_wait(sXA_cstate))
        mma.accumulating_gemm_ss(tidx_, tiled_mma_261, sXA, sLB, acc, sXA_cstate, sXA_cstate, acc_accumulate, -1)
        acc_accumulate = True
        cute.nvgpu.warpgroup.wait_group(0)
        sLB_pipe.consumer_release(sXA_cstate)
        sXA_cstate.advance()
        sLB_pstate.advance()
        store.mma_epilogue_tma(tiled_mma_282, c_tma_tensor_1, c_tma_atom_1, acc_epi_smem, acc, 128, 256, sched_coord[0], sched_coord[1], tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 512, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((8, 8), (2, 4)), ((32, 1), (256, 8)), (32, 16), 8)
          for k in cutlass.range(0, 64, 1):
            if cutlass.const_expr(True):
              sB_pipe.producer_acquire(sB_pstate, sB_pipe.producer_try_acquire(sB_pstate))
              mcast_mask_4, cta_coord_4, cta_layout_4 = shared.get_multicast_info(None, -1)
              shared.tma_copy(a_tma_atom_4, a_tma_tensor_4, sA, 128, 64, sched_coord[0], k, sB_pipe, sB_pstate, cta_coord_4, cta_layout_4, mcast_mask_4)
              mcast_mask_5, cta_coord_5, cta_layout_5 = shared.get_multicast_info(None, -1)
              shared.tma_copy(b_tma_atom_5, b_tma_tensor_5, sB, 256, 64, sched_coord[1], k, sB_pipe, sB_pstate, cta_coord_5, cta_layout_5, mcast_mask_5)
            sB_pstate.advance()
          if cutlass.const_expr(True):
            sLB_pipe.producer_acquire(sLB_pstate, sLB_pipe.producer_try_acquire(sLB_pstate))
            mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info(None, -1)
            shared.tma_copy(xa_tma_atom_2, xa_tma_tensor_2, sXA, 128, 16, sched_coord[0], 0, sLB_pipe, sLB_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
            mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info(None, -1)
            shared.tma_copy(lora_b_tma_atom_3, lora_b_tma_tensor_3, sLB, 256, 16, sched_coord[1], 0, sLB_pipe, sLB_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
          sLB_pstate.advance()