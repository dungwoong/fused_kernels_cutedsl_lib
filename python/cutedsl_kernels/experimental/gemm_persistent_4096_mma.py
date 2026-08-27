import cutlass
from cutlass import cute
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import scheduler
from cdsl_helpers import mma
from cdsl_helpers import mma_sm80
from cdsl_helpers import elementwise
from cdsl_helpers import store, test


class Kernel:
  @cute.jit
  def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    sA_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    sB_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 256, 64, 3)
    sC_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 32, 2)
    acc_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_180959759749188303650482690196456611119 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    tiled_mma_b = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 256, 128, False)
    tiled_mma_gemm = mma_sm80.get_tiled_mma((128 // 16, 1, 1))
    tiled_mma_148085055983264653138226893410544968444 = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    c_tma_atom_1, c_tma_tensor_1 = shared.get_tma_epi_tensor_and_atom(c, sC_layout, 128, 32)
    a_tma_atom_2, a_tma_tensor_2 = shared.get_tma_tensor_and_atom(a, sA_layout, 128, 64, 2)
    b_tma_atom_3, b_tma_tensor_3 = shared.get_tma_tensor_and_atom(b, sB_layout, 256, 64, 1)
    self.kernel(a, b, c, sA_layout, sB_layout, sC_layout, acc_tiled_mma, tiled_mma_180959759749188303650482690196456611119, tiled_mma_148085055983264653138226893410544968444, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3, tiled_mma_b, tiled_mma_gemm).launch(grid=[1, 2, 66], block=384, cluster=(1, 2, 1))

  @cute.kernel
  def kernel(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, sA_layout, sB_layout, sC_layout, acc_tiled_mma, tiled_mma_180959759749188303650482690196456611119, tiled_mma_148085055983264653138226893410544968444, c_tma_atom_1, c_tma_tensor_1, a_tma_atom_2, a_tma_tensor_2, b_tma_atom_3, b_tma_tensor_3, tiled_mma_b, tiled_mma_gemm):
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
        sched_coord_pre = scheduler.remap_1d_idx(sched_idx, ((8, 32), 1), ((32, 1), 256), (32, 8), 8)
        sched_coord = scheduler.add_cluster_offset_2d(sched_coord_pre, (1, 2, 1))
        acc = mma_sm80.get_acc(tiled_mma_gemm, cutlass.Float32)
        acc.fill(0.0)
        acc_accumulate = False
        for k in cutlass.range(0, 64, 1):
          pipe.consumer_wait(state_c, pipe.consumer_try_wait(state_c))

          # load a and b to registers
          # I wonder if old wgmma will be fine here
          a_regs = mma_sm80.copy_mma_bf16(tidx_, tiled_mma_gemm, sA[None, None, state_c.index], True)
          b_regs = mma_sm80.copy_mma_bf16(tidx_, tiled_mma_gemm, sB[None, None, state_c.index], False)
          mma_sm80.gemm_0(tiled_mma_gemm, acc, a_regs, b_regs)
          print('a_regs', a_regs)
          print('b_regs', b_regs)
          print('tiled_mma', tiled_mma_gemm)
          
          # No extra sync is required because release occurs after.
          pipe.consumer_release(state_c)
          state_c.advance()
        # tiled_mma_148085055983264653138226893410544968444
        store.mma_epilogue_tma(tiled_mma_gemm, c_tma_tensor_1, c_tma_atom_1, sC, acc, 128, 256, sched_coord[0], sched_coord[1], tidx_, warpidx_, cutlass.Float32)
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        # Producer
        state_p = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
        for sched_idx in cutlass.range(cute.arch.block_idx()[2], 256, 66):
          sched_coord_pre = scheduler.remap_1d_idx(sched_idx, ((8, 32), 1), ((32, 1), 256), (32, 8), 8)
          sched_coord = scheduler.add_cluster_offset_2d(sched_coord_pre, (1, 2, 1))
          for k in cutlass.range(0, 64, 1):
            pipe.producer_acquire(state_p, pipe.producer_try_acquire(state_p))
            mcast_mask_2, cta_coord_2, cta_layout_2 = shared.get_multicast_info((1, 2, 1), 1)
            shared.tma_copy(a_tma_atom_2, a_tma_tensor_2, sA, 128, 64, sched_coord[0], k, pipe, state_p, cta_coord_2, cta_layout_2, mcast_mask_2)
            mcast_mask_3, cta_coord_3, cta_layout_3 = shared.get_multicast_info((1, 2, 1), 0)
            shared.tma_copy(b_tma_atom_3, b_tma_tensor_3, sB, 256, 64, sched_coord[1], k, pipe, state_p, cta_coord_3, cta_layout_3, mcast_mask_3)
            state_p.advance()
        pipe.producer_tail(state_p)