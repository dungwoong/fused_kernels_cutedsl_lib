import torch
import time
from triton.testing import do_bench
from cutedsl_kernels.experimental.lora import Kernel as HelLora

from cdsl_helpers.cdsl_fn_utils import compile_cutedsl

def torch_lora(a, b, lA, lB):
    return (a @ b.t()) + (a @ lA.t() @ lB.t())

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
    parser.add_argument("--mode", type=int, choices=[0], default=0)
    args = parser.parse_args()

    m, n, k = args.m, args.n, args.k
    lora_dim = 16
    if args.mode in (1, 2):
        print('Overriding shape')
        m = n = k = 4096
    
    htype = torch.float64
    dtype = torch.bfloat16
    a64 = torch.randn((m, k), dtype=htype).to('cuda')
    b64 = torch.randn((n, k), dtype=htype).to('cuda')
    lA64 = torch.randn((lora_dim, k), dtype=htype).to('cuda')
    lB64 = torch.randn((n, lora_dim), dtype=htype).to('cuda')

    a = a64.to(dtype)
    b = b64.to(dtype)
    lA = lA64.to(dtype)
    lB = lB64.to(dtype)
    lxA = a @ lA.t()
    c = torch.empty((m, n), dtype=dtype).to('cuda')
    # ref = torch.nn.functional.relu(a @ b.t())
    ref64 = torch_lora(a64, b64, lA64, lB64)
    compiled_torch = torch.compile(torch_lora)
    ref = compiled_torch(a, b, lA, lB)

    print('Reference finished')
    gemm_classes = {
        0: HelLora,
    }
    GemmCls = gemm_classes[args.mode]

    gemm = GemmCls()
    print('Compiling kernel')
    compiled_gemm = compile_cutedsl((a, b, lxA, lB, c), gemm, False)

    def cdsl_fn(a, b, lA, lB):
        c_ = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device='cuda')
        lxA = a @ lA.t()
        compiled_gemm(a, b, lxA, lB, c_)
        return c_
    print('Running gemm')
    cdsl_ref = cdsl_fn(a, b, lA, lB)

    torch_rmse = get_rmse(ref64, ref.to(htype))
    my_rmse = get_rmse(ref64, cdsl_ref.to(htype))

    print(f'{torch_rmse=}, {my_rmse=}')

    torch_ms = do_bench(lambda: compiled_torch(a, b, lA, lB))
    time.sleep(2)
    my_ms = do_bench(lambda: cdsl_fn(a, b, lA, lB))
    print(f'{my_ms=}, {torch_ms=}, {torch_ms / my_ms}')