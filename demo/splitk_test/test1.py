import torch
from triton.testing import do_bench
from cutedsl_kernels.splitk_testing import SplitK1
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl
import time

torch.manual_seed(42)

if __name__ == '__main__':
    print('Starting...')
    a = torch.randn((4096, 4096), dtype=torch.bfloat16, device='cuda')
    b = torch.randn((16, 4096), dtype=torch.bfloat16, device='cuda')
    c = torch.zeros((4096, 16), dtype=torch.bfloat16, device='cuda')
    ref = a @ b.t()
    
    kernel = SplitK1(
        mnk=(128, 16, 64),
        stages=3,
        cluster_m=4,
        k_splits=4,
    )
    compiled_kernel = compile_cutedsl((a, b, c), kernel, False)

    compiled_kernel(a, b, c)
    torch.cuda.synchronize()

    # print(c)
    # print(ref)
    print(f'Max diff {(c - ref).max().item()}')

    def my_fn(a: torch.Tensor, b: torch.Tensor):
        c = torch.zeros((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device='cuda')
        compiled_kernel(a, b, c)
        return c

    my_ms = do_bench(lambda: my_fn(a, b))
    time.sleep(2)
    torch_ms = do_bench(lambda: a @ b.t())
    print(f'{my_ms=} {torch_ms=}, ({torch_ms/my_ms})')

    # Max diff 2.0
    # my_ms=0.023865938013425923 torch_ms=0.025302546254567876, (1.060194920490189)
    # As a next step, I want to see if I can reduce in DSMEM instead of GMEM using TMA store
    # since you lose precision