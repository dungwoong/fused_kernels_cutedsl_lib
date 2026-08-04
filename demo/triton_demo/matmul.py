import torch
from triton_kernels.persistent_matmul import matmul_descriptor_persistent, matmul_tma_persistent
from triton.testing import do_bench
import time

def get_rmse(ref: torch.Tensor, o: torch.Tensor):
    assert o.dtype == ref.dtype
    mse = torch.nn.functional.mse_loss(o, ref, reduction='mean')
    rmse = mse.sqrt().item()
    return rmse

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
    ref64 = a64 @ b64.t()

    a = a64.to(dtype)
    b = b64.to(dtype)
    ref = a @ b.t()

    print('Reference finished')
    print('Reference rmse:', get_rmse(ref.to(htype), ref64))
    triton_output_1 = matmul_descriptor_persistent(a, b, True)
    triton_output_2 = matmul_tma_persistent(a, b, True)
    triton_output_3 = matmul_descriptor_persistent(a, b, False)
    triton_output_4 = matmul_tma_persistent(a, b, False)
    print('Descriptor + WS rmse:', get_rmse(triton_output_1.to(htype), ref64))
    print('TMA + WS rmse:', get_rmse(triton_output_2.to(htype), ref64))
    print('Descriptor rmse:', get_rmse(triton_output_3.to(htype), ref64))
    print('TMA rmse:', get_rmse(triton_output_4.to(htype), ref64))
    
    @torch.compile
    def torch_fn(a, b):
        # return torch.nn.functional.relu(a @ b.t())
        return a @ b.t()

    desc_ms = do_bench(lambda: matmul_descriptor_persistent(a, b, False))
    time.sleep(2)
    tma_ms = do_bench(lambda: matmul_tma_persistent(a, b, False))
    time.sleep(2)
    desc_ms_ws = do_bench(lambda: matmul_descriptor_persistent(a, b, True))
    time.sleep(2)
    tma_ms_ws = do_bench(lambda: matmul_tma_persistent(a, b, True))
    time.sleep(2)
    torch_ms = do_bench(lambda: torch_fn(a, b))
    print(f'{desc_ms=}({torch_ms / desc_ms})\n{tma_ms=}({torch_ms / tma_ms})\n{desc_ms_ws=}({torch_ms/desc_ms_ws})\n{tma_ms_ws=}({torch_ms / tma_ms_ws})\n{torch_ms=}')

    results = {
        "matmul_descriptor_persistent": desc_ms,
        "matmul_tma_persistent": tma_ms,
        "matmul_descriptor_persistent (WS)": desc_ms_ws,
        "matmul_tma_persistent (WS)": tma_ms_ws,
    }

    fastest_method = min(results, key=results.get)
    print(f'Fastest: {fastest_method}')