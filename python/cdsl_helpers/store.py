import cutlass
from cutlass import cute
from typing import Type

def get_stmatrix(transpose: bool, num_matrices: cutlass.Int32, element_type: Type[cutlass.Numeric]):
    return cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=num_matrices),
        element_type,
    )