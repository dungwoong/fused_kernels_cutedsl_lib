import torch
from triton.testing import do_bench
from cutedsl_kernels import AttnSM90
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl
import time
import math

if __name__ == '__main__':
    print('Starting...')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["debug", "speed", "ncu"])
    parser.add_argument("b", type=int, default=4)
    parser.add_argument("h", type=int, default=32)
    parser.add_argument("l", type=int, default=4096)
    parser.add_argument("d", type=int, default=128)
    args = parser.parse_args()

    bs, h, seqlen, dim = args.b, args.h, args.l, args.d

    rt = 1 / math.sqrt(dim)
    q = torch.randn((bs, h, seqlen, dim), dtype=torch.bfloat16).add(0.5).to('cuda')
    k = torch.randn((bs, h, seqlen, dim), dtype=torch.bfloat16).add(0.5).to('cuda')
    v = torch.randn((bs, h, seqlen, dim), dtype=torch.bfloat16).add(0.5).to('cuda')
    o = torch.zeros((bs, h, seqlen, dim), dtype=torch.bfloat16).to('cuda')

    fa = AttnSM90(qk_mn=(128, 128), num_stages=2, cluster_size_m=1, intra_wg_overlap=False, pingpong=True, epi_n=32, epi_stages=4)
    compiled_fa = compile_cutedsl((q, k, v, o, rt), fa, False)
    compiled_fa(q, k, v, o, rt)