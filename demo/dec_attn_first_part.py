import torch
from triton.testing import do_bench
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
    tensors = (X, WQ, WK, WV, Q_out, K_cache, V_cache)

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
        pytorch_compiled = torch.compile(pytorch_fn)
        ms_torch = do_bench(pytorch_compiled)
        print(f'{ms_torch=}')
    