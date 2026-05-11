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
assumes N is 1
"""


@dataclass
class SplitKArguments(ArgumentsBase):
    """
    Assumes N is 1, so we just have m and we also split along K
    """
    cluster_size: cutlass.Int32
    nclusters_m: cutlass.Int32
    k_split: cutlass.Int32

    @staticmethod
    @cute.jit
    def create(cluster_size: cutlass.Int32, nclusters_m: cutlass.Int32, k_split: cutlass.Int32):
        return SplitKArguments(cluster_size, nclusters_m, k_split)


class SplitKScheduler:
    """
    Grid size will be cluster_size, nclusters_m * k_split
    So clusters come first, and then you can rasterize however you want

    Cluster works on a larger tile of the output, and they all should have the same k split
    """

    @dataclass
    class Params(ParamsBase):
        cluster_size: cutlass.Int32
        nclusters_m: cutlass.Int32
        k_split: cutlass.Int32
        cluster_size_divmod: FastDivmod

        @staticmethod
        @cute.jit
        def create(args: SplitKArguments, *, loc=None, ip=None):
            return SplitKScheduler.Params(
                cluster_size=args.cluster_size,
                nclusters_m=args.nclusters_m,
                k_split=args.k_split,
                cluster_size_divmod=FastDivmod.create(args.cluster_size)
            )

    def __init__(self, current_work_idx: Int32, num_tiles_executed: Int32, params: Params, *, loc=None, ip=None):
        self._current_work_idx = current_work_idx
        self._num_tiles_executed = num_tiles_executed
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: SplitKArguments, *, loc=None, ip=None) -> Params:
        return SplitKScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> SplitKScheduler:
        # NOTE this is hard-coded to support the cluster size since that's given in the params
        bidx, bidy, _ = cute.arch.block_idx()
        return SplitKScheduler(Int32(bidy * params.cluster_dim + bidx), Int32(0), params, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(params: Params, max_sms: Int32, *, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
        return (params.cluster_size, params.nclusters_m * params.k_split)

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
