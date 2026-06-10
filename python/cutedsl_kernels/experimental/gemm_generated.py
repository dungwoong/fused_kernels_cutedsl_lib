import cutlass
from cutlass import cute
from cdsl_helpers import shared
from cdsl_helpers import pipeline
from cdsl_helpers import mma
from cdsl_helpers import store


class Kernel:
  @cute.jit
  def __call__(self, a: cute.Tensor, b: cute.Tensor):
    sA_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 3)
    sB_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 256, 64, 3)
    acc_tiled_mma = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 256, False)
    a_tma_atom_1, a_tma_tensor_1 = shared.get_tma_tensor_and_atom(a, sA_layout, 128, 64)
    b_tma_atom_2, b_tma_tensor_2 = shared.get_tma_tensor_and_atom(b, sB_layout, 256, 64)
    self.kernel(sA_layout, sB_layout, acc_tiled_mma, a_tma_atom_1, a_tma_tensor_1, b_tma_atom_2, b_tma_tensor_2).launch(grid=[32, 16, 1], block=384)

  @cute.kernel
  def kernel(self, sA_layout, sB_layout, acc_tiled_mma, a_tma_atom_1, a_tma_tensor_1, b_tma_atom_2, b_tma_tensor_2):
    SharedStorage_t = shared.get_smem_struct()
    shared.smem_add_shared_tensor(SharedStorage_t, 'sA_ptr', cutlass.BFloat16, sA_layout, 1024)
    shared.smem_add_shared_tensor(SharedStorage_t, 'sB_ptr', cutlass.BFloat16, sB_layout, 1024)
    shared.smem_add_barrier_array(SharedStorage_t, 'pipe_ptr', 3)
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_ = smem_alloc.allocate(cute.struct(SharedStorage_t))
    sA = shared.smem_get_tensor(smem_, 'sA_ptr', sA_layout)
    sB = shared.smem_get_tensor(smem_, 'sB_ptr', sB_layout)
    pipe = pipeline.make_tma_pipeline_alt(smem_, 'pipe_ptr', 3, shared.staged_tensor_sizes(cutlass.BFloat16, sA_layout, sB_layout), 8)
    warpidx_ = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx_, _, _ = cute.arch.thread_idx()
    if warpidx_ >= 0 and warpidx_ < 8:
      cute.arch.setmaxregister_increase(232)
      # No change to min warp
      # Consumer
      state_c = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 3)
      acc = mma.get_acc(acc_tiled_mma, 128, 256, cutlass.Float32)
      acc_accumulate = False
      for k in cutlass.range(0, 64, 1):
        pipe.consumer_wait(state_c, pipe.consumer_try_wait(state_c))
        mma.accumulating_gemm_ss(tidx_, acc_tiled_mma, sA, sB, acc, state_c, state_c, acc_accumulate, 0)
        acc_accumulate = True
        pipe.consumer_release(state_c)
        state_c.advance()
      # TODO epilogue
      store.mma_epilogue_tma(acc_tiled_mma, "tma_tensor", "tma_atom", "shared_tensor", acc, )
    if warpidx_ >= 8 and warpidx_ < 12:
      cute.arch.setmaxregister_decrease(40)
      if warpidx_ == 8:
        warpidx_ = warpidx_ + 8
        tidx_ = tidx_ + 256
        # Producer
        state_p = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 3)
        for k in cutlass.range(0, 64, 1):
          pipe.producer_acquire(state_p, pipe.producer_try_acquire(state_p))
          shared.tma_copy(a_tma_atom_1, a_tma_tensor_1, sA, 128, 64, k, cute.arch.block_idx()[0], pipe, state_p)
          shared.tma_copy(b_tma_atom_2, b_tma_tensor_2, sB, 256, 64, k, cute.arch.block_idx()[1], pipe, state_p)
          state_p.advance()