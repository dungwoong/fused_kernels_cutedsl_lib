import torch
from triton.testing import do_bench
from cutedsl_kernels import DAttn2
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl
import time


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", action="store_true")
    args = parser.parse_args()
    IS_NCU = not args.speed # ncu doesn't accept -- I think
    print('Starting...')

    constants = {
        'M': 16,
        'D': 128,
        'N': 4096,
        'P': 1024,
        'H': 32,
    }
    M = 16
    D = 128
    N = 4096
    P = 1024
    H = 32
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    std = 0.01
    dtype = torch.bfloat16
    X = torch.randn((M, N), dtype=dtype).mul(std).to('cuda')
    WQ = torch.randn((N, N), dtype=dtype).mul(std).to('cuda')
    WK = torch.randn((N, N), dtype=dtype).mul(std).to('cuda')
    WV = torch.randn((N, N), dtype=dtype).mul(std).to('cuda')

    K_cache = torch.zeros((H, P, D), dtype=dtype).mul(std).to('cuda')
    V_cache = torch.zeros((H, P, D), dtype=dtype).mul(std).to('cuda')
    Q_out = torch.zeros((H, M, D), dtype=dtype).to('cuda')

    attn = DAttn2((16, 128, 64), 2, False)
    tensors = (X, WQ, WK, WV, Q_out, K_cache, V_cache)
    compiled_attn = compile_cutedsl(tensors, attn, False)
    compiled_attn(*tensors)
    torch.cuda.synchronize()

    K_cache_torch = torch.zeros_like(K_cache)
    V_cache_torch = torch.zeros_like(V_cache)
    W_qkv = torch.cat([WQ.t(), WK.t(), WV.t()], dim=1).to('cuda') # transpose due to how my cutedsl kernel is ran
    def pytorch_fn():
        qkv = torch.matmul(X, W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(M, H, D)
        k = k.view(M, H, D)
        v = v.view(M, H, D)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1) # H M D
        v = v.transpose(0, 1)
        K_cache_torch[:, P-M:, :] = k
        V_cache_torch[:, P-M:, :] = v
        return q
    
    q_torch = pytorch_fn()
    
    if not IS_NCU:
        print(q_torch.shape, Q_out.shape)
        k_cache_allclose = torch.allclose(K_cache, K_cache_torch)
        v_cache_allclose = torch.allclose(V_cache, V_cache_torch)
        q_allclose = torch.allclose(Q_out, q_torch)
        print(f'{k_cache_allclose=} {v_cache_allclose=} {q_allclose=}')
        # print(K_cache[0, P-M:, :])
        # print(K_cache_torch[0, P-M:, :])
        # print(X[:, 0])
        # print(X[:, 64])
        cutedsl_attn = lambda: compiled_attn(*tensors)
        pytorch_compiled = torch.compile(pytorch_fn)
        zeros_fn = lambda: torch.zeros((H, M, 3 * N), dtype=torch.bfloat16, device='cuda')
        ms_cdsl = do_bench(cutedsl_attn)
        time.sleep(2)
        ms_torch = do_bench(pytorch_compiled)
        time.sleep(1)
        ms_zeros = do_bench(zeros_fn)
        print(f'{ms_cdsl}, {ms_torch} ({ms_torch / ms_cdsl})')
        print(f'{ms_zeros=}')
    