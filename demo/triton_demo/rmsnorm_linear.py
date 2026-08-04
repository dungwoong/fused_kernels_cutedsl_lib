import torch
from triton_kernels.rmsnorm_linear import rmsnorm_linear_tma_persistent
from triton.testing import do_bench
import time

def get_rmse(ref: torch.Tensor, o: torch.Tensor):
    assert o.dtype == ref.dtype
    mse = torch.nn.functional.mse_loss(o, ref, reduction='mean')
    rmse = mse.sqrt().item()
    return rmse

def torch_kernel(a: torch.Tensor, b: torch.Tensor):
    a_rms = torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[1],), eps=1e-5)
    return a_rms @ b.t()

if __name__ == '__main__':
    print('Starting...')
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("mode", type=str, choices=["debug", "speed", "ncu"])
    parser.add_argument("m", type=int, default=4096)
    parser.add_argument("n", type=int, default=4096)
    parser.add_argument("k", type=int, default=4096)
    args = parser.parse_args()

    m, n, k = args.m, args.n, args.k
    
    htype = torch.float64
    dtype = torch.bfloat16
    a64 = torch.randn((m, k), dtype=htype).to('cuda')
    b64 = torch.randn((n, k), dtype=htype).to('cuda')
    ref64 = torch_kernel(a64, b64)

    a = a64.to(dtype)
    b = b64.to(dtype)
    ref = torch_kernel(a, b)

    print('Reference finished')
    print('Reference rmse:', get_rmse(ref.to(htype), ref64))
    triton_output_1 = rmsnorm_linear_tma_persistent(a, b, False)
    triton_output_2 = rmsnorm_linear_tma_persistent(a, b, True)
    print('TMA RMSE:', get_rmse(triton_output_1.to(htype), ref64))
    print('TMA + WS:', get_rmse(triton_output_2.to(htype), ref64))
    
    @torch.compile
    def torch_fn(a, b):
        # return torch.nn.functional.relu(a @ b.t())
        return a @ b.t()

    tma_ms = do_bench(lambda: rmsnorm_linear_tma_persistent(a, b, False))
    time.sleep(2)
    tma_ws_ms = do_bench(lambda: rmsnorm_linear_tma_persistent(a, b, True))
    time.sleep(2)
    torch_ms = do_bench(lambda: torch_fn(a, b))
    print(f'{tma_ms=}({torch_ms / tma_ms})\n{tma_ws_ms=}({torch_ms / tma_ws_ms})\n{torch_ms=}')
