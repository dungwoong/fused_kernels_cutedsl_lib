import torch
from triton.testing import do_bench
from cutedsl_kernels import DAttn1
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl
import time
import math

torch.manual_seed(52)

def get_rmse(ref: torch.Tensor, o: torch.Tensor):
    assert o.dtype == ref.dtype
    mse = torch.nn.functional.mse_loss(o, ref, reduction='mean')
    rmse = mse.sqrt().item()
    return rmse

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
    H = 32
    multiplier = D ** -0.5

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtype = torch.float64
    dtype = torch.bfloat16
    Q64 = torch.randn((H, M, D), dtype=rtype).to('cuda')
    K64 = torch.randn((H, P, D), dtype=rtype).to('cuda')
    V64 = torch.randn((H, D, P), dtype=rtype).to('cuda')
    Vt64 = V64.transpose(1, 2).contiguous()

    Q = Q64.to(dtype)
    K = K64.to(dtype)
    V = V64.to(dtype)
    Vt = Vt64.to(dtype)

    O = torch.empty((H, M, D), dtype=dtype).to('cuda')

    def torch_fn(Q_, K_, Vt_):
        P = (Q_ @ K_.transpose(1, 2)).mul(multiplier * math.log2(math.e))
        pre_softmax = torch.exp2(P)
        o = (pre_softmax @ Vt_)
        rowsum = torch.sum(pre_softmax, dim=-1)[..., None]
        return o * torch.reciprocal(rowsum)
    ref64 = torch_fn(Q64, K64, Vt64)
    ref = torch_fn(Q, K, Vt)
    # ref = Q @ K[:, -128:, :].transpose(1, 2)

    kernel = DAttn1(
        qk_mnk=(16, 128, 128),
        stages=2,
        p_stages=1,
        is_persistent=True
        )
    
    tensors = (Q, K, V, O, multiplier)
    compiled_attn = compile_cutedsl(tensors, kernel, False)
    compiled_attn(*tensors)
    torch.cuda.synchronize()

    # print(ref.shape, O.shape)
    print(ref[0, 0, :4])
    print(O[0, 0, :4])
    
    # print(O[0, :16, :16])

    if not IS_NCU:
        ref_rmse = get_rmse(ref64, ref.to(ref64.dtype))
        my_rmse = get_rmse(ref64, O.to(ref64.dtype))
        print(f'{ref_rmse=}, {my_rmse=}')
        # print(torch.sum(Q @ K.transpose(1, 2), axis=-1))
        print('max err', (ref - O).max().item())
        allclose = torch.allclose(ref, O)
        print(f'{allclose=}')
        compiled_torch = torch.compile(torch_fn)
        my_ms = do_bench(lambda: compiled_attn(*tensors))
        time.sleep(2)
        torch_ms = do_bench(lambda: compiled_torch(Q, K, Vt))
        print(f'{my_ms=}, {torch_ms=} ({torch_ms/my_ms})')

