import torch
from cutedsl_kernels.experimental.tma_test import Kernel
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl

if __name__ == '__main__':
    print('Starting...')
    # 1 won't print anything, 2 works though.
    a = torch.ones((3, 128), dtype=torch.bfloat16, device='cuda')

    kernel = Kernel()
    compiled_kernel = compile_cutedsl((a,), kernel, False)
    print('runnig kerne')
    compiled_kernel(a)
    torch.cuda.synchronize()
