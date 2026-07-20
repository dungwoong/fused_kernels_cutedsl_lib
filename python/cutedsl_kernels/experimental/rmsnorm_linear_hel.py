import cutlass
from cutlass import cute
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import scheduler
from cdsl_helpers import reduction
from cdsl_helpers import mma
from cdsl_helpers import elementwise
from cdsl_helpers import store

# kwargs={'tma_stages': 3}


class Kernel:
  @cute.jit
  def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    st_254_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    st_272_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 256, 64, 3)
    rt_310_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    tiled_mma_239 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, False)
    wgmma_acc_232_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_1669 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, True)
    tiled_mma_1718 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, True)
    tiled_mma_1738 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    c_tma_atom_1, c_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(c, rt_310_epi_smem_layout, 128, 64)
    a_tma_atom_2, a_tma_tensor_2 = shared.get_tma_tensor_and_atom(a, st_254_layout, 128, 64, 1)
    b_tma_atom_3, b_tma_tensor_3 = shared.get_tma_tensor_and_atom(b, st_272_layout, 256, 64, 1)
    self.kernel(st_254_layout, st_272_layout, a, b, c, rt_310_epi_smem_layout, tiled_mma_239, wgmma_acc_232_tiled_mma, tiled_mma_1669, tiled_mma_1718, tiled_mma_1738, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3).launch(grid=[132, 1, 1], block=384)

  @cute.kernel
  def kernel(self, st_254_layout, st_272_layout, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, rt_310_epi_smem_layout, tiled_mma_239, wgmma_acc_232_tiled_mma, tiled_mma_1669, tiled_mma_1718, tiled_mma_1738, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_254_ptr', cutlass.BFloat16, st_254_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_272_ptr', cutlass.BFloat16, st_272_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_272_pipe_ptr', 3)
    shared.smem_add_shared_tensor(SharedStorage_t, 'rt_310_epi_smem_ptr', cutlass.BFloat16, rt_310_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    st_254 = shared.smem_get_tensor(smem_, 'st_254_ptr', st_254_layout)
    st_272 = shared.smem_get_tensor(smem_, 'st_272_ptr', st_272_layout)
    st_272_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_272_pipe_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, st_272_layout, st_254_layout), 8, None, 1)
    rt_310_epi_smem = shared.smem_get_tensor(smem_, 'rt_310_epi_smem_ptr', rt_310_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    st_254_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_254_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_272_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_272_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 128, 132):
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((4, 16), (1, 2)), ((32, 1), (128, 16)), (32, 4), 4)
        rt_238 = reduction.make_mma_A_reduction_tensor(tiled_mma_239, 128, 16, cutlass.Float32)
        wgmma_acc_232 = mma.get_acc(wgmma_acc_232_tiled_mma, 128, 256, cutlass.Float32)
        wgmma_acc_232_accumulate = False
        for k in cutlass.range(0, 64, 1):
          st_272_pipe.consumer_wait(st_254_cstate, st_272_pipe.consumer_try_wait(st_254_cstate))
          rt_264 = mma.copy_a_wgmma(tidx_, tiled_mma_1669, st_254[None, None, st_254_cstate.index], 128, 64, cutlass.BFloat16)
          mma.accumulating_gemm_rs(tidx_, tiled_mma_1718, rt_264, st_272, wgmma_acc_232, st_254_cstate, wgmma_acc_232_accumulate, -1)
          wgmma_acc_232_accumulate = True
          reduction.row_sum_square_mixed_types(rt_264, rt_238, cutlass.BFloat16)
          cute.nvgpu.warpgroup.wait_group(0)
          st_272_pipe.consumer_release(st_254_cstate)
          st_254_cstate.advance()
          st_272_pstate.advance()
        rt_238.store(reduction.warp_sum_row_mma_layout(rt_238.load()))
        rt_291 = cute.make_rmem_tensor_like(rt_238, cutlass.Float32)
        rt_291.store(elementwise.const_div(rt_238.load(), 4096.0))
        rt_295 = cute.make_rmem_tensor_like(rt_291, cutlass.Float32)
        rt_295.store(elementwise.const_add(rt_291.load(), 1e-05))
        rt_299 = cute.make_rmem_tensor_like(rt_295, cutlass.Float32)
        rt_299.store(elementwise.const_rsqrt(rt_295.load()))
        rt_310 = elementwise.row_mul(wgmma_acc_232, rt_299)
        store.mma_epilogue_tma(tiled_mma_1738, c_tma_tensor_1, c_tma_atom_1, rt_310_epi_smem, rt_310, 128, 256, sched_coord[0], sched_coord[1], tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 128, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((4, 16), (1, 2)), ((32, 1), (128, 16)), (32, 4), 4)
          for k in cutlass.range(0, 64, 1):
            if cutlass.const_expr(True):
              st_272_pipe.producer_acquire(st_272_pstate, st_272_pipe.producer_try_acquire(st_272_pstate))
              mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info(None, -1)
              shared.tma_copy(a_tma_atom_2, a_tma_tensor_2, st_254, 128, 64, sched_coord[0], k, st_272_pipe, st_272_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info(None, -1)
              shared.tma_copy(b_tma_atom_3, b_tma_tensor_3, st_272, 256, 64, sched_coord[1], k, st_272_pipe, st_272_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
            st_272_pstate.advance()