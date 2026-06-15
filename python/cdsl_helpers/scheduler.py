import cutlass
from cutlass import cute

@cute.jit
def flip_indexes(i, group_size):
    slow, fast = i // group_size, i % group_size
    should_flip = slow % 2
    return slow * group_size + fast if not should_flip else slow * group_size + (group_size - 1 - fast)

@cute.jit
def remap_1d_idx(i: int, layout_shape: tuple, layout_stride: tuple, output_shape: tuple, group_size: int):
    """
    layout <layout_shape>:<layout_stride> must map onto a grid of the same
    size as the other layout <shape>, which is assumed to be colexicographically ordered
    """
    i = flip_indexes(i, group_size) if cutlass.const_expr(group_size is not None) else i
    l = cute.make_layout(layout_shape, stride=layout_stride)
    idx = l(i) # 1d offset 
    output_layout = cute.make_layout(output_shape)
    return output_layout.get_hier_coord(idx)
