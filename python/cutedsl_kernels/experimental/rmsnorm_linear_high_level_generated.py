import cutlass
from cutlass import cute
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import scheduler
from cdsl_helpers import mma
from cdsl_helpers import reduction
from cdsl_helpers import elementwise
from cdsl_helpers import store

# kwargs={'tma_stages': 3}


class Kernel:
  @cute.jit
  def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    sA_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    sB_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 256, 64, 3)
    acc_scaled_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    acc_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_320543768995522252220864416052355818781 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, False)
    tiled_mma_233708676893453240447061255376323023517 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, True)
    tiled_mma_83620134480117676418404359634357297195 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, True)
    tiled_mma_96742965177584133873450483786579412025 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    c_tma_atom_1, c_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(c, acc_scaled_epi_smem_layout, 128, 64)
    a_tma_atom_2, a_tma_tensor_2 = shared.get_tma_tensor_and_atom(a, sA_layout, 128, 64, 1)
    b_tma_atom_3, b_tma_tensor_3 = shared.get_tma_tensor_and_atom(b, sB_layout, 256, 64, 1)
    self.kernel(sA_layout, sB_layout, a, b, c, acc_scaled_epi_smem_layout, acc_tiled_mma, tiled_mma_320543768995522252220864416052355818781, tiled_mma_233708676893453240447061255376323023517, tiled_mma_83620134480117676418404359634357297195, tiled_mma_96742965177584133873450483786579412025, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3).launch(grid=[132, 1, 1], block=384, cluster=[1, 1, 1])

  @cute.kernel
  def kernel(self, sA_layout, sB_layout, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, acc_scaled_epi_smem_layout, acc_tiled_mma, tiled_mma_320543768995522252220864416052355818781, tiled_mma_233708676893453240447061255376323023517, tiled_mma_83620134480117676418404359634357297195, tiled_mma_96742965177584133873450483786579412025, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'sA_ptr', cutlass.BFloat16, sA_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sB_ptr', cutlass.BFloat16, sB_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'sB_pipe_ptr', 3)
    shared.smem_add_shared_tensor(SharedStorage_t, 'acc_scaled_epi_smem_ptr', cutlass.BFloat16, acc_scaled_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    sA = shared.smem_get_tensor(smem_, 'sA_ptr', sA_layout)
    sB = shared.smem_get_tensor(smem_, 'sB_ptr', sB_layout)
    sB_pipe = pipeline.make_tma_pipeline_alt(smem_, 'sB_pipe_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, sB_layout, sA_layout), 8, cute.make_layout((1, 1, 1, 1)), 1)
    acc_scaled_epi_smem = shared.smem_get_tensor(smem_, 'acc_scaled_epi_smem_ptr', acc_scaled_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    sA_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    sA_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    sB_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    sB_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 128, 132):
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((4, 32), 1), ((32, 1), 128), (32, 4), 4)
        acc = mma.get_acc(acc_tiled_mma, 128, 256, cutlass.Float32)
        acc_accumulate = False
        sum_acc = reduction.make_mma_A_reduction_tensor(tiled_mma_320543768995522252220864416052355818781, 128, 16, cutlass.Float32)
        for k in cutlass.range(0, 64, 1):
          sB_pipe.consumer_wait(sA_cstate, sB_pipe.consumer_try_wait(sA_cstate))
          a_regs = mma.copy_a_wgmma(tidx_, tiled_mma_233708676893453240447061255376323023517, sA[None, None, sA_cstate.index], 128, 64, cutlass.BFloat16)
          mma.accumulating_gemm_rs(tidx_, tiled_mma_83620134480117676418404359634357297195, a_regs, sB, acc, sA_cstate, acc_accumulate, -1)
          acc_accumulate = True
          reduction.row_sum_square_mixed_types(a_regs, sum_acc, cutlass.BFloat16)
          cute.nvgpu.warpgroup.wait_group(0)
          sB_pipe.consumer_release(sA_cstate)
          sA_cstate.advance()
          sB_pstate.advance()
        warpreduced_acc = cute.make_rmem_tensor_like(sum_acc, cutlass.Float32)
        warpreduced_acc.store(reduction.warp_sum_row_mma_layout(sum_acc.load()))
        sumacc_0 = cute.make_rmem_tensor_like(warpreduced_acc, cutlass.Float32)
        sumacc_0.store(elementwise.const_div(warpreduced_acc.load(), 4096))
        sumacc_1 = cute.make_rmem_tensor_like(sumacc_0, cutlass.Float32)
        sumacc_1.store(elementwise.const_add(sumacc_0.load(), 1e-05))
        sumacc_2 = cute.make_rmem_tensor_like(sumacc_1, cutlass.Float32)
        sumacc_2.store(elementwise.const_rsqrt(sumacc_1.load()))
        acc_scaled = elementwise.row_mul(acc, sumacc_2)
        store.mma_epilogue_tma(tiled_mma_96742965177584133873450483786579412025, c_tma_tensor_1, c_tma_atom_1, acc_scaled_epi_smem, acc_scaled, 128, 256, sched_coord[0], sched_coord[1], tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 128, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((4, 32), 1), ((32, 1), 128), (32, 4), 4)
          for k in cutlass.range(0, 64, 1):
            if cutlass.const_expr(True):
              sB_pipe.producer_acquire(sB_pstate, sB_pipe.producer_try_acquire(sB_pstate))
              mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info([1, 1, 1], -1)
              shared.tma_copy(a_tma_atom_2, a_tma_tensor_2, sA, 128, 64, sched_coord[0], k, sB_pipe, sB_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info([1, 1, 1], -1)
              shared.tma_copy(b_tma_atom_3, b_tma_tensor_3, sB, 256, 64, sched_coord[1], k, sB_pipe, sB_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
            sB_pstate.advance()