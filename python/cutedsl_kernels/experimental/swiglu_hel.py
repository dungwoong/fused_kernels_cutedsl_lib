import cutlass
from cutlass import cute
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import scheduler
from cdsl_helpers import mma
from cdsl_helpers import elementwise
from cdsl_helpers import store

# kwargs={'tma_stages': 3}


class Kernel:
  @cute.jit
  def __call__(self, a: cute.Tensor, b: cute.Tensor, b1: cute.Tensor, c: cute.Tensor):
    st_1943_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    st_1957_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    st_1978_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    rt_1995_epi_smem_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    wgmma_acc_1916_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    wgmma_acc_1922_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    tiled_mma_6663 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 16, True)
    tiled_mma_6712 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, True)
    tiled_mma_6767 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, True)
    tiled_mma_6787 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128, False)
    c_tma_atom_1, c_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(c, rt_1995_epi_smem_layout, 128, 64)
    a_tma_atom_2, a_tma_tensor_2 = shared.get_tma_tensor_and_atom(a, st_1943_layout, 128, 64, 1)
    b_tma_atom_3, b_tma_tensor_3 = shared.get_tma_tensor_and_atom(b, st_1957_layout, 128, 64, 1)
    b1_tma_atom_4, b1_tma_tensor_4 = shared.get_tma_tensor_and_atom(b1, st_1978_layout, 128, 64, 1)
    self.kernel(st_1943_layout, st_1957_layout, st_1978_layout, a, b, b1, c, rt_1995_epi_smem_layout, wgmma_acc_1916_tiled_mma, wgmma_acc_1922_tiled_mma, tiled_mma_6663, tiled_mma_6712, tiled_mma_6767, tiled_mma_6787, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3, b1_tma_atom_4, b1_tma_tensor_4).launch(grid=[132, 1, 1], block=384)

  @cute.kernel
  def kernel(self, st_1943_layout, st_1957_layout, st_1978_layout, a: cute.Tensor, b: cute.Tensor, b1: cute.Tensor, c: cute.Tensor, rt_1995_epi_smem_layout, wgmma_acc_1916_tiled_mma, wgmma_acc_1922_tiled_mma, tiled_mma_6663, tiled_mma_6712, tiled_mma_6767, tiled_mma_6787, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3, b1_tma_atom_4, b1_tma_tensor_4):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_1943_ptr', cutlass.BFloat16, st_1943_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_1957_ptr', cutlass.BFloat16, st_1957_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'st_1978_ptr', cutlass.BFloat16, st_1978_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'st_1978_pipe_ptr', 3)
    shared.smem_add_shared_tensor(SharedStorage_t, 'rt_1995_epi_smem_ptr', cutlass.BFloat16, rt_1995_epi_smem_layout, 1024)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    st_1943 = shared.smem_get_tensor(smem_, 'st_1943_ptr', st_1943_layout)
    st_1957 = shared.smem_get_tensor(smem_, 'st_1957_ptr', st_1957_layout)
    st_1978 = shared.smem_get_tensor(smem_, 'st_1978_ptr', st_1978_layout)
    st_1978_pipe = pipeline.make_tma_pipeline_alt(smem_, 'st_1978_pipe_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, st_1978_layout, st_1957_layout, st_1943_layout), 8, None, 1)
    rt_1995_epi_smem = shared.smem_get_tensor(smem_, 'rt_1995_epi_smem_ptr', rt_1995_epi_smem_layout)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    st_1943_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_1943_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_1957_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_1957_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    st_1978_pstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
    st_1978_cstate = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      for sched_idx in cutlass.range(cute.arch.block_idx()[0], 1024, 132):
        sched_coord = scheduler.remap_1d_idx(sched_idx, ((4, 16), (8, 2)), ((32, 1), (128, 16)), (32, 32), 4)
        wgmma_acc_1916 = mma.get_acc(wgmma_acc_1916_tiled_mma, 128, 128, cutlass.Float32)
        wgmma_acc_1916_accumulate = False
        wgmma_acc_1922 = mma.get_acc(wgmma_acc_1922_tiled_mma, 128, 128, cutlass.Float32)
        wgmma_acc_1922_accumulate = False
        for k in cutlass.range(0, 64, 1):
          st_1978_pipe.consumer_wait(st_1943_cstate, st_1978_pipe.consumer_try_wait(st_1943_cstate))
          rt_1953 = mma.copy_a_wgmma(tidx_, tiled_mma_6663, st_1943[None, None, st_1943_cstate.index], 128, 64, cutlass.BFloat16)
          mma.accumulating_gemm_rs(tidx_, tiled_mma_6712, rt_1953, st_1957, wgmma_acc_1916, st_1943_cstate, wgmma_acc_1916_accumulate, -1)
          wgmma_acc_1916_accumulate = True
          mma.accumulating_gemm_rs(tidx_, tiled_mma_6767, rt_1953, st_1978, wgmma_acc_1922, st_1943_cstate, wgmma_acc_1922_accumulate, -1)
          wgmma_acc_1922_accumulate = True
          cute.nvgpu.warpgroup.wait_group(0)
          st_1978_pipe.consumer_release(st_1943_cstate)
          st_1943_cstate.advance()
          st_1978_pstate.advance()
        rt_1974 = cute.make_rmem_tensor_like(wgmma_acc_1916, cutlass.Float32)
        rt_1974.store(elementwise.silu(wgmma_acc_1916.load()))
        rt_1995 = elementwise.tilewise_mul(rt_1974, wgmma_acc_1922)
        store.mma_epilogue_tma(tiled_mma_6787, c_tma_tensor_1, c_tma_atom_1, rt_1995_epi_smem, rt_1995, 128, 128, sched_coord[0], sched_coord[1], tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        for sched_idx in cutlass.range(cute.arch.block_idx()[0], 1024, 132):
          sched_coord = scheduler.remap_1d_idx(sched_idx, ((4, 16), (8, 2)), ((32, 1), (128, 16)), (32, 32), 4)
          for k in cutlass.range(0, 64, 1):
            if cutlass.const_expr(True):
              st_1978_pipe.producer_acquire(st_1978_pstate, st_1978_pipe.producer_try_acquire(st_1978_pstate))
              mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info(None, -1)
              shared.tma_copy(a_tma_atom_2, a_tma_tensor_2, st_1943, 128, 64, sched_coord[0], k, st_1978_pipe, st_1978_pstate, cta_coord_2, cta_layout_2, mcast_mask_2)
              mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info(None, -1)
              shared.tma_copy(b_tma_atom_3, b_tma_tensor_3, st_1957, 128, 64, sched_coord[1], k, st_1978_pipe, st_1978_pstate, cta_coord_3, cta_layout_3, mcast_mask_3)
              mcast_mask_4, cta_coord_4, cta_layout_4 = shared.get_multicast_info(None, -1)
              shared.tma_copy(b1_tma_atom_4, b1_tma_tensor_4, st_1978, 128, 64, sched_coord[1], k, st_1978_pipe, st_1978_pstate, cta_coord_4, cta_layout_4, mcast_mask_4)
            st_1978_pstate.advance()