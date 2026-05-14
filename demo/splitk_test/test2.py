import torch
from triton.testing import do_bench
from cutedsl_kernels.splitk_testing import SplitK2, ReduceDowncastKernel
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl
import time

torch.manual_seed(42)

def get_rmse(ref: torch.Tensor, o: torch.Tensor):
    assert o.dtype == ref.dtype
    mse = torch.nn.functional.mse_loss(o, ref, reduction='mean')
    rmse = mse.sqrt().item()
    return rmse

if __name__ == '__main__':
    with torch.no_grad():
        print('Starting...')
        NCU = False
        SPLITS = 2
        m, n, k = 4096, 16, 4096
        a64 = torch.randn((m, k), dtype=torch.float64)
        b64 = torch.randn((n, k), dtype=torch.float64)
        a = a64.to(dtype=torch.bfloat16)
        b = b64.to(dtype=torch.bfloat16)
        a64 = a64.to('cuda')
        b64 = b64.to('cuda')
        a = a.to('cuda')
        b = b.to('cuda')
        c = torch.empty((m, SPLITS, n), dtype=torch.float32, device='cuda')
        o = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        ref = a64 @ b64.t()
        torch_ref = a @ b.t()
        torch.cuda.synchronize()

        # This beats on mnk 8192 16 8192 with SPLITS=1, just need big K size
        # kernel = SplitK2(
        #     mnk=(64, 16, 128),
        #     stages=4,
        #     cluster_m=2,
        #     k_splits=SPLITS,
        # )
        kernel = SplitK2(
            mnk=(64, 16, 128),
            stages=4,
            cluster_m=2,
            k_splits=SPLITS,
        )
        # m, n, splits
        reduce_kernel = ReduceDowncastKernel(
            32,
            16,
            SPLITS,
        )
        compiled_kernel = compile_cutedsl((a, b, c), kernel, False)
        compiled_kernel(a, b, c)

        compiled_reduce = compile_cutedsl((c, o), reduce_kernel, False)
        compiled_reduce(c, o)
        # print(c)
        torch.cuda.synchronize()

        @torch.compile
        def matmul(a, b):
            return a @ b.t()

        # NOTE I had this as finished = x.sum(...) and torch was optimizing out the entire op
        @torch.compile
        def combine(x):
            return x.sum(axis=1).to(torch.bfloat16)

        def my_fn(a: torch.Tensor, b: torch.Tensor):
            c = torch.empty((a.shape[0], SPLITS, b.shape[0]), dtype=torch.float32, device='cuda')
            o = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device='cuda')
            compiled_kernel(a, b, c)
            compiled_reduce(c, o)
            return o
        
        my_answer = my_fn(a, b)

        # print(c)
        # print(ref)
        if not NCU:

            diff = torch.abs(my_answer - torch_ref)
            print(f'nnz: {torch.count_nonzero(diff)}')
            rmse_mine = get_rmse(ref, my_answer.to(torch.float64))
            rmse_torch = get_rmse(ref, torch_ref.to(torch.float64))
            print(f'{rmse_mine=}, {rmse_torch=}')
            # print(f'allclose: {torch.allclose(o, torch_ref)}')

            my_ms = do_bench(lambda: my_fn(a, b))
            time.sleep(2)
            torch_ms = do_bench(lambda: matmul(a, b))
            time.sleep(2)
            combine_ms = do_bench(lambda: compiled_reduce(c, o))
            combine_ms_torch = do_bench(lambda: combine(c))
            print(f'{my_ms=} {torch_ms=}, ({torch_ms/my_ms})')
            print(f'{combine_ms=} {combine_ms_torch=}, ({combine_ms_torch / combine_ms})')
            # print(f'{torch_ms / (my_ms + combine_ms)}')

# rmse_mine=0.18454818691927724, rmse_torch=0.1845493359157738
# my_ms=0.024110091747630628 torch_ms=0.02518035176368865, (1.0443905409925784)
# combine_ms=0.005804218710905098 combine_ms_torch=0.005650011791018788, (0.9734319246797881)
# these numbers are acceptable