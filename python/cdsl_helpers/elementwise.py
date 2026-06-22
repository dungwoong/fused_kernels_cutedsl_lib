
import math
from typing import Callable, Type, Union, Optional
import cutlass
from cutlass import cute, const_expr, Int32, Boolean
from cutlass.cute.nvgpu import warpgroup
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cutlass_dsl import Numeric, dsl_user_op, T
from cutlass.utils import LayoutEnum
from cutlass._mlir.dialects import nvvm, llvm, arith

@dsl_user_op
def _fmax(a: float | cutlass.Float32, b: float | cutlass.Float32, c: float | cutlass.Float32 | None = None, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        nvvm.fmax(
            T.f32(),
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            c=cutlass.Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )

@cute.jit
def _relu(x: cute.TensorSSA | cutlass.Float32) -> cute.TensorSSA | cutlass.Float32:
    if const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_rmem_tensor(x.shape, cutlass.Float32)
        for i in cutlass.range_constexpr(cute.size(x.shape)):
            res[i] = _fmax(x[i], 0)
        return res.load()
    else:
        return _fmax(x, 0)

@cute.jit
def relu_f32(x: cute.Tensor):
    assert x.element_type == cutlass.Float32, "relu only supports fp32 input"
    out = cute.make_rmem_tensor_like(x, cutlass.Float32)
    out.store(_relu(x.load()))
    return out

@cute.jit
def const_div(x: cute.TensorSSA, val: int) -> cute.TensorSSA:
    out = cute.make_rmem_tensor(x.shape, x.dtype)
    for i in cutlass.range_constexpr(cute.size(x.shape)):
        out[i] = x[i] / val
    return out.load()

@cute.jit
def const_add(x: cute.TensorSSA, val: int) -> cute.TensorSSA:
    out = cute.make_rmem_tensor(x.shape, x.dtype)
    for i in cutlass.range_constexpr(cute.size(x.shape)):
        out[i] = x[i] + val
    return out.load()

@cute.jit
def const_rsqrt(x: cute.TensorSSA) -> cute.TensorSSA:
    out = cute.make_rmem_tensor(x.shape, x.dtype)
    for i in cutlass.range_constexpr(cute.size(x.shape)):
        out[i] = cute.math.rsqrt(x[i], fastmath=True)
    return out.load()