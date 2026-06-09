# Quick test to make sure scripts still work as intended

echo GEMM expect 1.01x
python3 demo/gemm_run.py speed 4096 4096 4096

echo RMSNORM LINEAR expect 1.07x
python3 demo/rmsnorm_linear.py speed 4096 4096 4096

echo RMSNORM SWIGLU expect fast ver speedup RS wgmma 1.12x
python3 demo/rmsnorm_swiglu.py speed 4096 4096 4096

echo SWIGLU expect RS wgmma 1.10x
python3 demo/swiglu.py speed 4096 4096 4096

echo DEC ATTN TOGETHER expect 1.50x sdpa
python3 demo/dec_attn/dec_attn_second_part.py 1024 --speed

echo DEC ATTN SPLITK expect 1.32x sdpa
python3 demo/dec_attn/splitk.py 1024 --speed

echo MISSING LORA test