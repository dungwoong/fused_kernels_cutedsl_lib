import torch
import time
from triton.testing import do_bench
from cutedsl_kernels.experimental.gemm_generated import Kernel
from cutedsl_kernels.experimental.gemm_persistent_4096 import Kernel as GemmPersistent
from cutedsl_kernels.experimental.gemm_high_level_generated import Kernel as GemmHL
from cutedsl_kernels.experimental.gemm_hel import Kernel as GemmHel
from cutedsl_kernels.experimental.gemm_hel_3072 import Kernel as Gemm3072
from cutedsl_kernels.experimental.gemm_persistent_4096_mma import Kernel as Gemm4096Mma

# Experimental autogen
from cutedsl_kernels.experimental_autogen.gemm_c2_mnk4096_1 import Kernel as GemmC2Mnk4096_1
from cutedsl_kernels.experimental_autogen.gemm_c2_mnk4096_2 import Kernel as GemmC2Mnk4096_2

from cdsl_helpers.cdsl_fn_utils import compile_cutedsl


if __name__ == '__main__':
    print('Starting...')
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("mode", type=str, choices=["debug", "speed", "ncu"])
    parser.add_argument("m", type=int, default=4096)
    parser.add_argument("n", type=int, default=4096)
    parser.add_argument("k", type=int, default=4096)
    parser.add_argument("--mode", type=int, choices=[0, 1, 2, 3, 4, 5], default=0)
    args = parser.parse_args()

    m, n, k = args.m, args.n, args.k
    if args.mode in (1, 2, 5):
        print('Overriding shape')
        m = n = k = 4096
    if args.mode == 4:
        print('Overriding shape')
        m = n = k = 3072
    
    a = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    # ref = torch.nn.functional.relu(a @ b.t())
    ref = a @ b.t()

    print('Reference finished')
    gemm_classes = {
        0: GemmHL, # default, custom user-specified option
        1: GemmC2Mnk4096_1,
        2: GemmC2Mnk4096_2,
        3: GemmHel,
        4: Gemm3072,
        5: Gemm4096Mma,
    }
    GemmCls = gemm_classes[args.mode]

    gemm = GemmCls()
    print('Compiling kernel')
    compiled_gemm = compile_cutedsl((a, b, c), gemm, False)
    print('Running gemm')
    compiled_gemm(a, b, c)
    print(c - ref)
    print(c)
    # print(c)
    print('allclose:', torch.allclose(ref, c))

    def gemm_fn(a, b):
        c_ = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a, b, c_)
    
    @torch.compile
    def torch_fn(a, b):
        # return torch.nn.functional.relu(a @ b.t())
        return a @ b.t()

    my_ms = do_bench(lambda: gemm_fn(a, b))
    time.sleep(2)
    torch_ms = do_bench(lambda: torch_fn(a, b))
    print(f'{my_ms=}, {torch_ms=}, {torch_ms / my_ms}')