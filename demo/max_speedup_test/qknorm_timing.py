import torch
from triton.testing import do_bench
from cutedsl_kernels import RMSNormLinear2SM90, RMSNormLinear1SM90
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl, STREAM
import time

def randn(*shape):
    return torch.randn(shape, dtype=torch.bfloat16, device='cuda')

if __name__ == '__main__':
    M = 16
    D = 128
    N = 4096
    P = 1024
    H = 32

    X = randn(M, N)
    Wqkv = randn(N, N)
    q = randn(H, M, D)
    k = randn(H, M, D)
    cache_K = randn(H, P, D)
    cache_V = randn(H, P, D)

    # matmul
    @torch.compile
    def qkv_mm():
        return X @ Wqkv
    
    @torch.compile
    def norm():
        q_var = q.pow(2).mean(-1, keepdim=True)
        k_var = k.pow(2).mean(-1, keepdim=True)
        q_norm = q * torch.rsqrt(q_var)
        k_norm = k * torch.rsqrt(k_var)
        return q_norm, k_norm
    
    @torch.compile
    def attn():
        scores = torch.matmul(q, cache_K.transpose(1, 2))
        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum

        output = torch.matmul(weights, cache_V)
        return output
    
    qkv_ms = do_bench(qkv_mm)
    norm_ms = do_bench(norm)
    attn_ms = do_bench(attn)
    total = qkv_ms + norm_ms + attn_ms
    print(f"{qkv_ms=} ({qkv_ms/total}) ({total / (total - qkv_ms)})")
    print(f"{norm_ms=}, ({norm_ms/total}) ({total / (total - norm_ms)})")
    print(f"{attn_ms=}, ({attn_ms/total}) ({total / (total - attn_ms)})")
