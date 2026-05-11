from typing import Tuple
import enum

import cutlass
from cutlass import cute, pipeline
from cdsl_helpers import shared, mma, pipeline as my_pipeline, layout as my_layout, store as my_store

@cute.jit
def print0(x):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        if tidx == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.print_tensor(x)
    else:
        if tidx == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.printf(x)


def get_epi_tensor_atom(t: cute.Tensor, epi_smem_layout_staged: cute.ComposedLayout, epi_tile: Tuple[int, int]):
    """
    This only works if you want a single stage
    and your epi SMEM layout has a single stage.
    """
    epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
    epi_tma_tensor_layout = cute.composition(cute.make_identity_layout(t.shape), epi_tile)
    op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
    tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        op, t, epi_smem_layout, epi_tma_tensor_layout
    )
    return tma_atom, tma_tensor


class Kernel:
    """
    Performs m16n128k64 matmul
    try TransposeA
    """
    def __init__(self):
        self.dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.m, self.n, self.k = (16, 64, 128)
        self.nconsumer_warps = None
        self.consumer_regs, self.producer_regs = 232, 40
        self.stages = 2
    
    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mO: cute.Tensor):
        # mA = my_layout.select(mA, [1, 0]) # (128, 16)
        # Attempt1: load in A and transpose the layout
        sA_layout = shared.get_smem_layout_row_major(self.dtype, self.k, self.m, self.stages)
        sB_layout = shared.get_smem_layout_row_major(self.dtype, self.n, self.k, self.stages)
        sO_layout = shared.get_smem_layout_row_major(self.dtype, self.m, self.n, 1)

        tiled_gemm = mma.get_tiled_mma(self.dtype, False, True, self.acc_dtype, self.m, self.n)
        consumer_wgs = tiled_gemm.size // 128
        self.nconsumer_warps = consumer_wgs * 4

        mA_atom, mA_tensor = shared.get_tma_tensor_and_atom(mA, sA_layout, self.m, self.k)
        mB_atom, mB_tensor = shared.get_tma_tensor_and_atom(mB, sB_layout, self.n, self.k)
        mO_atom, mO_tensor = get_epi_tensor_atom(mO, sO_layout, (self.m, self.n))
        
        self.kernel(
            sA_layout, sB_layout, sO_layout,
            mA_atom, mB_atom,
            mA_tensor, mB_tensor,
        ).launch(grid=1, block=[(self.nconsumer_warps + 4) * cute.arch.WARP_SIZE])

    @cute.kernel
    def kernel(
        self, sA_layout, sB_layout, sO_layout,
        atom_A, atom_B,
        mA, mB):
        # get SMEM
        # for range(hardcoded)
        #   load A
        #   load B
        #   matmul
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        SharedStorage = type("SS", (), dict())
        items = [
            ('sA_ptr', shared.memrange(self.dtype, sA_layout, 1024)),
            ('sB_ptr', shared.memrange(self.dtype, sB_layout, 1024)),
            ('sO_ptr', shared.memrange(self.dtype, sO_layout, 1024)),
            ('pipe_ptr', cute.struct.MemRange[cutlass.Int64, self.stages * 2]), 
        ]
        for k, v in items:
            SharedStorage.__annotations__[k] = v
        
        s_alloc = cutlass.utils.SmemAllocator()
        smem = s_alloc.allocate(cute.struct(SharedStorage))
        sA = shared.smem_get_tensor(smem, 'sA_ptr', sA_layout)
        sB = shared.smem_get_tensor(smem, 'sB_ptr', sB_layout)
        sO = shared.smem_get_tensor(smem, 'sO_ptr', sO_layout)

        a_bytes = cute.size_in_bytes(self.dtype, cute.select(sA_layout, mode=[0, 1]))
        b_bytes = cute.size_in_bytes(self.dtype, cute.select(sB_layout, mode=[0, 1]))
        pipe = my_pipeline.make_tma_pipeline(
            smem.pipe_ptr.data_ptr(),
            self.stages,
            num_consumer_warps=self.nconsumer_warps,
            num_bytes=a_bytes + b_bytes,
            mcast_size=1,
            cta_layout_vmnk=None,
        )

        if (warp_idx < self.nconsumer_warps):
            cute.arch.setmaxregister_increase(self.consumer_regs)
            cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            for k in cutlass.range(1, unroll=1):
                pipe.consumer_wait(cstate, pipe.consumer_try_wait(cstate))
                pipe.consumer_release(cstate)
                cstate.advance()
        if (warp_idx >= self.nconsumer_warps):
            cute.arch.setmaxregister_decrease(self.producer_regs)
            if (warp_idx == self.nconsumer_warps):
                pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
                for k in cutlass.range(1, unroll=1):
                    pipe.producer_acquire(pstate, pipe.producer_try_acquire(pstate))
                    shared.tma_copy(atom_A, mA, sA, self.m, self.k, 0, k, pipe, pstate)
                    shared.tma_copy(atom_B, mB, sB, self.n, self.k, 0, k, pipe, pstate)
                    pstate.advance()