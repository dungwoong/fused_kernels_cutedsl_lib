import torch
import time
from triton.testing import do_bench
from cutedsl_kernels.experimental.attn import Kernel as Attn
from cutedsl_kernels.experimental.attn_full import Kernel as AttnFull
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl
from torch.nn.attention import SDPBackend, sdpa_kernel
from contextlib import nullcontext

def get_rmse(ref: torch.Tensor, o: torch.Tensor):
    assert o.dtype == ref.dtype
    mse = torch.nn.functional.mse_loss(o, ref, reduction='mean')
    rmse = mse.sqrt().item()
    return rmse

if __name__ == '__main__':
    print('Starting...')
    nheads, q_len, kv_len, dim = 32, 4096, 4096, 128
    
    htype = torch.float64
    dtype = torch.bfloat16
    q64 = torch.randn((nheads, q_len, dim), dtype=htype, device='cuda')
    k64 = torch.randn((nheads, kv_len, dim), dtype=htype, device='cuda')
    v64 = torch.randn((nheads, kv_len, dim), dtype=htype, device='cuda')

    q = q64.to(dtype)
    k = k64.to(dtype)
    v = v64.to(dtype)
    o = torch.empty((nheads, q_len, dim), dtype=dtype, device='cuda')

    ref64 = torch.nn.functional.scaled_dot_product_attention(q64.unsqueeze(0), k64.unsqueeze(0), v64.unsqueeze(0))
    ref = torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))

    attn = AttnFull()
    attn_compiled = compile_cutedsl((q, k, v, o), attn, False)
    print('Compiled kernel')
    attn_compiled(q, k, v, o)

    print(o[0, ...])
    print(ref[0, 0, ...])

    torch_rmse = get_rmse(ref64.squeeze(0), ref.squeeze(0).to(htype))
    my_rmse = get_rmse(ref64.squeeze(0), o.to(htype))
    print(f'{torch_rmse=}, {my_rmse=}')

    def cdsl_fn(q, k, v):
        o_ = torch.empty_like(q)
        attn_compiled(q, k, v, o_)
    
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
    # with nullcontext():
        my_ms = do_bench(lambda: cdsl_fn(q, k, v))
        time.sleep(2)
        torch_ms = do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)))
    print(f'{my_ms=}, {torch_ms=}, {torch_ms / my_ms}')