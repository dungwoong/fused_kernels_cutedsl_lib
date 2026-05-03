import torch
from triton.testing import do_bench
from cutedsl_kernels import DAttn1
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl
import time

torch.manual_seed(52)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", action="store_true")
    args = parser.parse_args()
    IS_NCU = not args.speed # ncu doesn't accept -- I think
    print('Starting...')

    M = 16
    D = 128
    N = 4096
    P = 1024
    H = 128

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    dtype = torch.bfloat16
    Q = torch.randn((H, M, D), dtype=dtype).to('cuda')
    K = torch.randn((H, P, D), dtype=dtype).to('cuda')
    V = torch.randn((H, D, P), dtype=dtype).to('cuda')
    Vt = V.transpose(1, 2).contiguous()
    O = torch.empty((H, M, D), dtype=dtype).to('cuda')

    def torch_fn():
        return (Q @ K.transpose(1, 2)) @ Vt
    ref = torch_fn()
    # ref = Q @ K[:, -128:, :].transpose(1, 2)

    kernel = DAttn1(
        qk_mnk=(16, 128, 128),
        stages=2,
        p_stages=1,
        is_persistent=True
        )
    
    tensors = (Q, K, V, O, 1.0)
    compiled_attn = compile_cutedsl(tensors, kernel, False)
    compiled_attn(*tensors)
    torch.cuda.synchronize()

    # print(ref.shape, O.shape)
    # print(ref[0, :16, :16])
    # print(O[0, :16, :16])
    
    # print(O[0, :16, :16])

    if not IS_NCU:
        print('max err', (ref - O).max().item())
        allclose = torch.allclose(ref, O)
        print(f'{allclose=}')
        compiled_torch = torch.compile(torch_fn)
        my_ms = do_bench(lambda: compiled_attn(*tensors))
        time.sleep(2)
        torch_ms = do_bench(compiled_torch)
        print(f'{my_ms=}, {torch_ms=} ({torch_ms/my_ms})')

