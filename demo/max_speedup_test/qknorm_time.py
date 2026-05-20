import torch
from triton.testing import do_bench
import time

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("seqlen", type=int, default=16)

    args = parser.parse_args()
    seqlen = args.seqlen
    # H seqlen dim
    x = torch.randn((seqlen, 32, 128), dtype=torch.bfloat16, device='cuda')
    x_reshape = x.view(seqlen, -1)
    w = torch.randn((4096, 4096), dtype=torch.bfloat16, device='cuda')

    @torch.compile
    def fn(a, b):
        o = a @ b.t()
        o = o.view(seqlen, 32, 128)
        o = torch.nn.functional.rms_norm(o, normalized_shape=(o.shape[-1],), eps=1e-5)
        # print(o.shape)
        return o
    
    @torch.compile
    def matmul(a, b):
        return a @ b.t()
    # fn(x_reshape, w)

    fn_ms = do_bench(lambda: fn(x_reshape, w))
    time.sleep(2)
    mm_ms = do_bench(lambda: matmul(x_reshape, w))

    print(f'{fn_ms=} {mm_ms=} {fn_ms / mm_ms}')
    print(f'rms is {1 - (mm_ms / fn_ms)} of the comp')
