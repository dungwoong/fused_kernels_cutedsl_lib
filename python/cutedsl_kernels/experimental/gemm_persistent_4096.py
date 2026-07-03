import cutlass
from cutlass import cute
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import scheduler
from cdsl_helpers import mma
from cdsl_helpers import elementwise
from cdsl_helpers import store


class Kernel:
  @cute.jit
  def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    sA_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    sB_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 256, 64, 3)
    sC_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 32, 2)
    acc_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    c_tma_atom_1, c_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(c, sC_layout, 128, 32)
    a_tma_atom_2, a_tma_tensor_2 = shared.get_tma_tensor_and_atom(a, sA_layout, 128, 64, 2)
    b_tma_atom_3, b_tma_tensor_3 = shared.get_tma_tensor_and_atom(b, sB_layout, 256, 64, 1)
    self.kernel(a, b, c, sA_layout, sB_layout, sC_layout, acc_tiled_mma, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3).launch(grid=[1, 2, 66], block=384, cluster=(1, 2, 1))

  @cute.kernel
  def kernel(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, sA_layout, sB_layout, sC_layout, acc_tiled_mma, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'sA_ptr', cutlass.BFloat16, sA_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sB_ptr', cutlass.BFloat16, sB_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sC_ptr', cutlass.BFloat16, sC_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'pipe_ptr', 3)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    sA = shared.smem_get_tensor(smem_, 'sA_ptr', sA_layout)
    sB = shared.smem_get_tensor(smem_, 'sB_ptr', sB_layout)
    sC = shared.smem_get_tensor(smem_, 'sC_ptr', sC_layout)
    pipe = pipeline.make_tma_pipeline_alt(smem_, 'pipe_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, sA_layout, sB_layout), 8, cute.make_layout((1, 1, 2, 1)), 2)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      # Consumer
      state_c = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
      for sched_idx in cutlass.range(cute.arch.block_idx()[2], 256, 66):
        sched_coord_pre = scheduler.remap_1d_idx(sched_idx, ((8, 32), 2), ((32, 1), 256), (32, 8), 8)
        sched_coord = scheduler.add_cluster_offset_2d(sched_coord_pre, (1, 2, 1))
        acc = mma.get_acc(acc_tiled_mma, 128, 256, cutlass.Float32)
        acc_accumulate = False
        for k in cutlass.range(0, 64, 1):
          pipe.consumer_wait(state_c, pipe.consumer_try_wait(state_c))
          mma.accumulating_gemm_ss(tidx_, acc_tiled_mma, sA, sB, acc, state_c, state_c, acc_accumulate, 0)
          acc_accumulate = True
          pipe.consumer_release(state_c)
          state_c.advance()
        newacc = elementwise.relu_f32(acc)
        store.mma_epilogue_tma(acc_tiled_mma, c_tma_tensor_1, c_tma_atom_1, sC, acc, 128, 256, sched_coord[0], sched_coord[1], tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        # Producer
        state_p = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
        for sched_idx in cutlass.range(cute.arch.block_idx()[2], 256, 66):
          sched_coord_pre = scheduler.remap_1d_idx(sched_idx, ((8, 32), 2), ((32, 1), 256), (32, 8), 8)
          sched_coord = scheduler.add_cluster_offset_2d(sched_coord_pre, (1, 2, 1))
          for k in cutlass.range(0, 64, 1):
            pipe.producer_acquire(state_p, pipe.producer_try_acquire(state_p))
            mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info((1, 2, 1), 1)
            shared.tma_copy(a_tma_atom_2, a_tma_tensor_2, sA, 128, 64, sched_coord[0], k, pipe, state_p, cta_coord_2, cta_layout_2, mcast_mask_2)
            mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info((1, 2, 1), 0)
            shared.tma_copy(b_tma_atom_3, b_tma_tensor_3, sB, 256, 64, sched_coord[1], k, pipe, state_p, cta_coord_3, cta_layout_3, mcast_mask_3)
            state_p.advance()
        pipe.producer_tail(state_p)