import torch
from triton.testing import do_bench
from cutedsl_kernels.splitk_testing import SplitK1
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl
import time

torch.manual_seed(42)

def get_rmse(ref: torch.Tensor, o: torch.Tensor):
    assert o.dtype == ref.dtype
    mse = torch.nn.functional.mse_loss(o, ref, reduction='mean')
    rmse = mse.sqrt().item()
    return rmse

if __name__ == '__main__':
    print('Starting...')
    NCU = True
    a64 = torch.randn((4096, 4096), dtype=torch.float64)
    b64 = torch.randn((16, 4096), dtype=torch.float64)
    a = a64.to(dtype=torch.bfloat16)
    b = b64.to(dtype=torch.bfloat16)
    a64 = a64.to('cuda')
    b64 = b64.to('cuda')
    a = a.to('cuda')
    b = b.to('cuda')
    c = torch.zeros((4096, 16), dtype=torch.bfloat16, device='cuda')
    ref = a64 @ b64.t()
    torch_ref = a @ b.t()
    
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
    if not NCU:
        rmse_mine = get_rmse(ref, c.to(torch.float64))
        rmse_torch = get_rmse(ref, torch_ref.to(torch.float64))
        print(f'{rmse_mine=}, {rmse_torch=}')

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