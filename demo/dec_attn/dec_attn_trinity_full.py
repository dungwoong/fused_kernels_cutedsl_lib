import torch
from triton.testing import do_bench
from triton_kernels.vanilla_trinity import forward as trinity_fwd
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl
from torch.nn.attention import SDPBackend, sdpa_kernel
from contextlib import nullcontext
import time
import math

torch.manual_seed(52)

def get_rmse(ref: torch.Tensor, o: torch.Tensor):
    assert o.dtype == ref.dtype
    mse = torch.nn.functional.mse_loss(o, ref, reduction='mean')
    rmse = mse.sqrt().item()
    return rmse

@torch.compile
def fill_cache(x, wQKV, cache_K, cache_V, P, M, H, D):
    """
    M: new seq len(16)
    P: old seqlen(1024)
    H: heads(32)
    D: dim(128)
    cache_K and V are (H, P, D)
    X is (M, N)
    W_qkv is (N, N3) so qkv are each (M, H, D)

    returns q(H, M, D), cache_K(H, P, D), cache_V(H, P, D)
    """
    N = M * H # model dim
    qkv = torch.matmul(x, wQKV)
    q, k, v = torch.chunk(qkv, 3, dim=-1)
    q = q.view(M, H, D)
    k = k.view(M, H, D)
    v = v.view(M, H, D)

    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    cache_K[:, P-M:P, :] = k
    cache_V[:, P-M:P, :] = v
    q = q.transpose(0, 1)
    return q, cache_K, cache_V

@torch.compile
def torch_impl(x, wQKV, cache_K, cache_V, P, M, H, D):
    q, cache_K, cache_V = fill_cache(x, wQKV, cache_K, cache_V, P, M, H, D)
    # BHSD
    output = torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(0), cache_K.unsqueeze(0), cache_V.unsqueeze(0))
    output = output.squeeze(0)
    return output.permute(1, 0, 2).contiguous().view(M, H * D)

# ncu --set full -o dec_attn -f --kernel-name "regex:^.*cutedsl.*" python3 demo/dec_attn_second_part.py
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("p", type=int)
    parser.add_argument("--speed", action="store_true")
    args = parser.parse_args()
    IS_NCU = not args.speed # ncu doesn't accept -- I think
    print('Starting...')

    M = 16
    D = 128
    # P = args.p
    P = 1024
    H = 32
    N = H * D
    print(f'{N=}')
    multiplier = D ** -0.5

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rtype = torch.float64
    dtype = torch.float16
    X64 = torch.randn((M, N), dtype=rtype).to('cuda')
    WQ64 = torch.randn((N, N), dtype=rtype).to('cuda')
    WK64 = torch.randn((N, N), dtype=rtype).to('cuda')
    WV64 = torch.randn((N, N), dtype=rtype).to('cuda')
    cache_K64 = torch.randn((H, P, D), dtype=rtype).to('cuda')
    cache_V64 = torch.randn((H, P, D), dtype=rtype).to('cuda')
    WQKV64 = torch.cat([WQ64, WK64, WV64], dim=1).to('cuda')

    X = X64.to(dtype)
    WQ = WQ64.to(dtype)
    WK = WK64.to(dtype)
    WV = WV64.to(dtype)
    WQKV = WQKV64.to(dtype)
    cache_K = cache_K64.to(dtype)
    cache_V = cache_V64.to(dtype)

    O_trinity = torch.empty((M, N), dtype=rtype).to('cuda')
    O = torch.empty((H, M, D), dtype=dtype).to('cuda')
    
    ref64 = torch_impl(X64, WQKV64, cache_K64, cache_V64, P, M, H, D)
    ref = torch_impl(X, WQKV, cache_K, cache_V, P, M, H, D)
    cache_K[:, P-M:P, :] = 0
    cache_V[:, P-M:P, :] = 0
    trinity_fwd(cache_K, O_trinity, cache_V, WK, WQ, WV, X)

    ref_rmse = get_rmse(ref64, ref.to(ref64.dtype))
    trinity_rmse = get_rmse(ref64, O_trinity.to(ref64.dtype))
    print(f'{ref_rmse=}, {trinity_rmse=}')
    print(O_trinity)

