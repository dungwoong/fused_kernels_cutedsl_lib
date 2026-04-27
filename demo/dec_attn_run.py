import torch
from cutedsl_kernels import DAttn1
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl

if __name__ == '__main__':
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

    attn = DAttn1((16, 128, 64), 2, False)
    tensors = (X, WQ, WK, WV, K_cache, V_cache)
    compiled_attn = compile_cutedsl(tensors, attn, False)
    compiled_attn(*tensors)
    torch.cuda.synchronize()

    K_cache_torch = torch.zeros_like(K_cache)
    V_cache_torch = torch.zeros_like(V_cache)
    W_qkv = torch.cat([WQ.t(), WK.t(), WV.t()], dim=1).to('cuda') # transpose due to how my cutedsl kernel is ran
    qkv = torch.matmul(X, W_qkv)
    q, k, v = torch.chunk(qkv, 3, dim=-1)
    q = q.view(M, H, D)
    k = k.view(M, H, D)
    v = v.view(M, H, D)
    k = k.transpose(0, 1) # H M D
    v = v.transpose(0, 1)

    K_cache_torch[:, P-M:, :] = k
    V_cache_torch[:, P-M:, :] = v
    k_cache_allclose = torch.allclose(K_cache, K_cache_torch)
    print(f'K allclose {k_cache_allclose}')
    # print(K_cache[0, P-M:, :])
    # print(K_cache_torch[0, P-M:, :])
    # print(X[:, 0])
    # print(X[:, 64])