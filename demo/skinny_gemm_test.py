import torch
from triton.testing import do_bench

# @torch.compile
def gemm(a: torch.Tensor, b: torch.Tensor):
    return a @ b.t()

if __name__ == '__main__':
    """
    The kernel to do the m16n128 gemm will automatically do the transpose.
    I profiled with ncu, both use wgmma.
    """
    m, n, k = 16, 4096, 4096
    a = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')

    gemm(a, b)
    gemm(b, a)
    time_ms_1 = do_bench(lambda: gemm(a, b))
    time_ms_2 = do_bench(lambda: gemm(b, a))
    print(f"{time_ms_1=}, {time_ms_2=}")