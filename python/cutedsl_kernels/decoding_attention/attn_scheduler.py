from __future__ import annotations
from typing import Tuple
from cdsl_helpers.utils.cute_dsl_utils import ParamsBase, ArgumentsBase
from cdsl_helpers.utils.fast_math import FastDivmod
from dataclasses import dataclass
import math

import cutlass
from cutlass import cute
from cutlass import Int32, const_expr

MAX_SM_H100 = 132

"""
Schedule heads to different blocks.
In the future, we can schedule start and end K for a split-k setup
"""


@dataclass
class HeadAttnTileSchedulerArguments(ArgumentsBase):
    """
    You assume that each block 
    will handle the entire sequence of new tokens
    so you only need to split along heads.

    No batch dim(mirroring Trinity Vanilla Attn)
    """
    nheads: cute.Shape
    is_persistent: cutlass.Constexpr[bool]

    @staticmethod
    @cute.jit
    def create(nheads: cutlass.Int32, is_persistent: bool):
        # last dim of t is the heads dim
        # assert cute.rank(t) == 3, "need 3d tensor, last dim is head"
        # nheads = cute.size(t, mode=[2])
        return HeadAttnTileSchedulerArguments(nheads, is_persistent)


class HeadAttnTileScheduler:

    @dataclass
    class Params(ParamsBase):
        n_heads: cutlass.Int32
        is_persistent: cutlass.Constexpr[bool]
        # n_heads_divmod: FastDivmod

        @staticmethod
        @cute.jit
        def create(args: HeadAttnTileSchedulerArguments, *, loc=None, ip=None):
            return HeadAttnTileScheduler.Params(
                n_heads=args.nheads,
                is_persistent=args.is_persistent,
                # n_heads_divmod=FastDivmod.create(args.nheads),
            )

    def __init__(self, current_work_idx: Int32, num_tiles_executed: Int32, params: Params, *, loc=None, ip=None):
        self._current_work_idx = current_work_idx
        self._num_tiles_executed = num_tiles_executed
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: HeadAttnTileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return HeadAttnTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> HeadAttnTileScheduler:
        # both rasterization schemes put all blocks in bidx
        bidx, _, _ = cute.arch.block_idx()
        return HeadAttnTileScheduler(Int32(bidx), Int32(0), params, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(params: Params, max_sms: Int32, *, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
        if const_expr(not params.is_persistent):
            return (math.prod(params.n_heads), 1, 1)
        else:
            return (max_sms, 1, 1)

    @cute.jit
    def _map_cta_coords(self, work_id: Int32, *, loc=None, ip=None) -> tuple[Int32, Int32]:
        # work_id is just the head to work on for now
        return work_id

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None):
        params = self.params
        head = self._map_cta_coords(self._current_work_idx)
        tile_coord_mnkl = (head, head, None, head) # if you put 0 here it gets optimized out for compile-time, and you get problems with dimension of tile coord
        is_valid = False
        if const_expr(not params.is_persistent):
            is_valid = self._num_tiles_executed == 0
        else:
            is_valid = self._current_work_idx < cute.size(params.n_heads)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    @cute.jit
    def fetch_next_work(self, *, loc=None, ip=None):
        pass

    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.is_persistent):
            num_persistent_ctas = cute.arch.grid_dim()[0]
            self._current_work_idx += Int32(num_persistent_ctas)

        self._num_tiles_executed += Int32(1)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self._current_work_idx,
            self._num_tiles_executed,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self._current_work_idx,
                self._num_tiles_executed,
                self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)
