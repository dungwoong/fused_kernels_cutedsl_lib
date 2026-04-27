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

    K_cache = torch.randn((H, P, D), dtype=dtype).mul(std).to('cuda')
    V_cache = torch.randn((H, P, D), dtype=dtype).mul(std).to('cuda')

    attn = DAttn1((16, 128, 64), 2, True)
    tensors = (X, WQ, WK, WV, K_cache, V_cache)
    compiled_attn = compile_cutedsl(tensors, attn, False)
    compiled_attn(*tensors)