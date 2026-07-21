
import math
from typing import Callable, Type, Union, Optional
import cutlass
from cutlass import cute, const_expr, Int32, Boolean
from cutlass.cute.nvgpu import warpgroup
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cutlass_dsl import Numeric, dsl_user_op, T
from cutlass.utils import LayoutEnum
from cutlass._mlir.dialects import nvvm, llvm, arith
from . import layout as my_layout, _quack_activation

"""
This also contains broadcast ops like elementwise div
"""

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
def const_div(x: cute.TensorSSA, val: float) -> cute.TensorSSA:
    out = cute.make_rmem_tensor(x.shape, x.dtype)
    for i in cutlass.range_constexpr(cute.size(x.shape)):
        out[i] = x[i] / val
    return out.load()

@cute.jit
def const_mul(x: cute.TensorSSA, val: float) -> cute.TensorSSA:
    out = cute.make_rmem_tensor(x.shape, x.dtype)
    for i in cutlass.range_constexpr(cute.size(x.shape)):
        out[i] = x[i] * val
    return out.load()

@cute.jit
def const_add(x: cute.TensorSSA, val: float) -> cute.TensorSSA:
    out = cute.make_rmem_tensor(x.shape, x.dtype)
    for i in cutlass.range_constexpr(cute.size(x.shape)):
        out[i] = x[i] + val
    return out.load()

@cute.jit
def const_mul_add(x: cute.TensorSSA, mul_val: float, add_val: float) -> cute.TensorSSA:
    """This is unnecessary if you do elementwise.mul then elementwise.add"""
    out = cute.make_rmem_tensor(x.shape, x.dtype)
    for i in cutlass.range_constexpr(cute.size(x.shape)):
        out[i] = (x[i] * mul_val) + add_val
    return out.load()

@cute.jit
def const_rsqrt(x: cute.TensorSSA) -> cute.TensorSSA:
    out = cute.make_rmem_tensor(x.shape, x.dtype)
    for i in cutlass.range_constexpr(cute.size(x.shape)):
        out[i] = cute.math.rsqrt(x[i], fastmath=True)
    return out.load()

@cute.jit
def row_mul(acc: cute.Tensor, scaler: cute.Tensor):
    new_acc = cute.make_rmem_tensor_like(acc, acc.element_type)
    a_mn = my_layout.make_acc_tensor_mn_view(acc, False)
    new_acc_mn = my_layout.make_acc_tensor_mn_view(new_acc, False)
    for r in cutlass.range_constexpr(cute.size(scaler)):
        new_acc_mn[r, None].store(a_mn[r, None].load() * scaler[r])
    return new_acc

@cute.jit
def row_bcast_sub(acc: cute.Tensor, vec: cute.Tensor):
    new_acc = cute.make_rmem_tensor_like(acc, acc.element_type)
    a_mn = my_layout.make_acc_tensor_mn_view(acc, False)
    new_acc_mn = my_layout.make_acc_tensor_mn_view(new_acc, False)
    for r in cutlass.range_constexpr(cute.size(vec)):
        new_acc_mn[r, None].store(a_mn[r, None].load() - vec[r])
    return new_acc

@cute.jit
def silu(acc: cute.TensorSSA):
    new_acc = cute.make_rmem_tensor_like(acc, acc.element_type)
    for r in cutlass.range_constexpr(cute.size(new_acc)):
        new_acc[r] = _quack_activation.silu(acc[r])
    return new_acc.load()

@cute.jit
def rcp(acc: cute.TensorSSA) -> cute.TensorSSA:
    new_acc = cute.make_rmem_tensor_like(acc, acc.element_type)
    for r in cutlass.range_constexpr(cute.size(new_acc)):
        new_acc[r] = cute.arch.rcp_approx(acc[r])
    return new_acc.load()

@cute.jit
def copy_elemwise(acc: cute.TensorSSA) -> cute.TensorSSA:
    return acc

@cute.jit
def exp2f(acc: cute.TensorSSA) -> cute.TensorSSA:
    new_acc = cute.make_rmem_tensor_like(acc, acc.element_type)
    for r in cutlass.range_constexpr(cute.size(new_acc)):
        new_acc[r] = cute.math.exp2(acc[r], fastmath=True)
    return new_acc.load()

@cute.jit
def tilewise_mul(a: cute.Tensor, b: cute.Tensor): # assume to have same layout
    new_acc = cute.make_rmem_tensor_like(a, a.element_type)
    for r in cutlass.range_constexpr(cute.size(new_acc)):
        new_acc[r] = a[r] * b[r]
    return new_acc

@cute.jit
def tilewise_sub(a: cute.Tensor, b: cute.Tensor): # assume to have same layout
    # returns A - B
    new_acc = cute.make_rmem_tensor_like(a, a.element_type)
    for r in cutlass.range_constexpr(cute.size(new_acc)):
        new_acc[r] = a[r] - b[r]
    return new_acc