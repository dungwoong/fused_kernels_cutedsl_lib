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

@cute.jit
def add_cluster_offset_2d(coord, cluster_shape):
    """
    NOTE: we could just do the layout to accommodate for clusters
    but this is an alternative.
    """
    assert cute.rank(coord) == 2
    assert len(cluster_shape) == 3 and cluster_shape[2] == 1
    cid_m, cid_n = coord[0], coord[1]
    bidx_in_cluster = cute.arch.block_in_cluster_idx()
    pid_m = cid_m * cluster_shape[0] + bidx_in_cluster[0]
    pid_n = cid_n * cluster_shape[1] + bidx_in_cluster[1]
    return (pid_m, pid_n)