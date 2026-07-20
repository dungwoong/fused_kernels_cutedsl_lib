# Can TMA copy stuff properly let's see with 1 block
import cutlass
from cutlass import cute
from cdsl_helpers import shared, pipeline
from cdsl_helpers.test import print0

class Kernel:
    @cute.jit
    def __call__(self, a: cute.Tensor):
        sA_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 64, 128, 1)
        a_tma_atom, a_tma_tensor = shared.get_tma_tensor_and_atom(a, sA_layout, 64, 128) # TODO ??
        self.kernel(a, sA_layout, a_tma_atom, a_tma_tensor).launch(grid=[1, 1, 1], block=256)
    
    @cute.kernel
    def kernel(self, a: cute.Tensor, sA_layout, a_tma_atom, a_tma_tensor):
        SS_t = shared.get_smem_struct()
        shared.smem_add_shared_tensor(SS_t, 'sA_ptr', cutlass.BFloat16, sA_layout, 1024)
        shared.smem_add_barrier_array(SS_t, 'pipe_ptr', 1)
        salloc = cutlass.utils.SmemAllocator()
        smem_ = salloc.allocate(cute.struct(SS_t))
        sA = shared.smem_get_tensor(smem_, 'sA_ptr', sA_layout)
        pipe = pipeline.make_tma_pipeline_alt(smem_, 'pipe_ptr', 1, shared.staged_tensor_sizes(cutlass.BFloat16, sA_layout), 4)
        warpidx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        print0('ENTERING WS')
        if warpidx < 4:
            cute.arch.setmaxregister_increase(232)
            state_c = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
            pipe.consumer_wait(state_c)
            print0("hi")
            print0(sA)
        
        if warpidx >= 4:
            cute.arch.setmaxregister_decrease(40)
            if warpidx == 4:
                state_p = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
                pipe.producer_acquire(state_p)
                shared.tma_copy(a_tma_atom, a_tma_tensor, sA, 64, 128, cute.arch.block_idx()[0], cute.arch.block_idx()[1], pipe, state_p)
                state_p.advance()