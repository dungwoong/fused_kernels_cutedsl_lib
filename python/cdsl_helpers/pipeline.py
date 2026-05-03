from cutlass import cute, pipeline
from dataclasses import dataclass
from cutlass.pipeline import PipelineTmaAsync as PipelineTmaAsyncOg, PipelineState
from typing import Optional
from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import if_generate

@dataclass(frozen=True)
class PipelineTmaAsync(PipelineTmaAsyncOg):
    """
    Override producer_acquire to take in extra_tx_count parameter.
    """

    @staticmethod
    def create(*args, **kwargs):
        obj = PipelineTmaAsyncOg.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        # obj.__class__ = PipelineTmaAsync
        object.__setattr__(obj, "__class__", PipelineTmaAsync)
        return obj

    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        extra_tx_count: int = 0,
        loc=None,
        ip=None,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        if const_expr(extra_tx_count == 0):
            self.sync_object_full.arrive(state.index, self.producer_mask)
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            self.sync_object_full.arrive_and_expect_tx(state.index, tx_count)


@cute.jit
def make_tma_pipeline(
    mbar_ptr: cute.Pointer, num_stages: int, num_bytes: int, cta_layout_vmnk: cute.Layout, mcast_size: int, num_consumer_warps: int
) -> pipeline.PipelineAsync:
    num_producers = 1
    num_consumers = num_consumer_warps * mcast_size  # IMPORTANT!!!

    producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_producers)
    consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_consumers)
    # NOTE: CTA layout is only used for syncing
    # This used to be pipeline.PipelineTmaAsync
    # but why not always support extra_tx
    return PipelineTmaAsync.create(
        barrier_storage=mbar_ptr,
        num_stages=num_stages,
        tx_count=num_bytes,
        producer_group=producer_group,
        consumer_group=consumer_group,
        cta_layout_vmnk=cta_layout_vmnk,
    )
