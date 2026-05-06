## first step is changing it from smem to register MMA
- Do you need a proxy fence after? [I don't think so](https://forums.developer.nvidia.com/t/why-arent-there-explicit-async-proxy-generic-proxy-fences-in-the-cuda-guide-tma-prefetching-example/357574)
- We can look at sm120 stuff too, since they have to use ldmatrix I think
- Doing a bunch of stuff with retiling etc. was good, since we're just addressing the A matrix here. In the future we might have to copy quack
- I took out async MMA but we'll see what we can pipeline later

## Adding warp reduction
```python
a_regs : tensor<ptr<bf16, rmem, align<32>> o ((8,1),1,4):((1,0),0,8)> # goes into the copy
a_regs0 : tensor<ptr<bf16, rmem, align<32>> o ((2,2,2),1,4):((1,2,4),0,8)> # is used for the WGMMA, 4 stages
```
- I manually made row reduce layout
- The row reduce algorithm is also quite manual too, it's just a 3-nested loop

# Modifying the epilogue
`accumulators : tensor<ptr<f32, rmem, align<32>> o ((2,2,32),1,1):((1,2,4),0,0)>`
- finished adding the elementwise rsqrt and sum to the sum, and then you just do a multiplication with each element in the accumulators

## Results
```
my_ms=0.19207999855279922, other_ms=0.2006400004029274
my_flops=715.5297506638651, other_flops=685.0027571570655
FLOPs numbers are incorrect since I only used the GEMMs flops but honestly RMS flops are nothing compared to GEMM

max_incorrect : 2.0
max_rel_incorrect : 26.5
```
- For some test randn matrices, we have ~40000 elements that were had an absolute error of >1. 1 is crazy work
- I should maybe double-check that all my stuff is correct, make sure sum is broadcasting properly. Otherwise, this is kinda what we would expect.
- I thought casting to fp32 would help precision though, but I'm guessing torch rmsnorm or whatever also casts to fp32

Next steps, figure out what to do about precision and then I can probably first test other kernels before working on optimization. RMSNorm seems like a bad candidate.

Also, anything I can do to make my work generalize better since I'm just manually hacking layouts at this point

## New RMSNorm+Linear
- If you try to do WGMMA in SMEM, you will spill. This is potentially causing a Misaligned Address error since now you're trying to get your reduction from local memory instead of other memory spaces.
- I don't know why, even if I do a constant multiplication at the end it's fine, but multiplying by the scale var makes things slow for some reason... from 18.4ms(gemm is around that speed for 4096) to 20.9 which is slower than gemm + whatever...

```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'kernel_cutlass_kernel_cutedsl_kernelsrmsnorm_linearattempt1gemm2GemmSM90_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypebf16_CopyAtom_ThrID10_TVLayoutSrc116384_0' for 'sm_90a'
ptxas info    : Function properties for kernel_cutlass_kernel_cutedsl_kernelsrmsnorm_linearattempt1gemm2GemmSM90_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypebf16_CopyAtom_ThrID10_TVLayoutSrc116384_0
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 168 registers, used 2 barriers
ptxas info    : Compile time = 107.803 ms
```
- ok it's because the multiplication at the end requires that intermediate summation to stay instead of getting optimized out. So we have to do something about that middle sum now.
- as a minor fuckup, since A is actually bf16 and the reduction is FP32, I was doing the square in BF16. Gotta add .to(type) and then square and then you can use fma instructions instead of mul.bf16 then add.f32

## Pingpong
- I don't even know what I was thinking with this one, there's just barely a little more float computation vs tensor core so you get to 90% compute so all this additional sync is not good.
- implementing ping-pong need to check the wgs are correct since attn had producer first

## Horizontal
- So each WG only needs to do half of the summation everytime, and then WGs will combine at the end.
- We could try a horizontal bcast and cluster reduce but it might lead to unwanted syncs, I feel like that wouldn't be worth it.
- To get the sum, you could have every thread sum what it needs or you could have one thread compute it and then shfl_sync. I'd say since warps are like SIMD I think we could just have every thread sum what it needs...?

## Split up the summation, use SMEM at the end
- horizontal tile so each WG does half the sum
- first, we should test whether GEMM is still performant like that.
- It might not be good since horizontal tile is like 10% slower. We could try clusters

I could also test how precise bf16 sum then FP32 thread accum or something would be. I could do that first since it seems easier

I could even first do the partial sum in BF16, then cast to FP32 then accumulate in FP32.

- ok accumulating in BF16 instead of FP32 gives a boost from 91% TCore usage to 96%. So now we drop from 0.193ms to 0.182ms getting up to 1.11x and we're still below the torch RMSE(0.18414 --> 0.18438)
- I tested having cluster shape as (1, 2, 1) for some reason that seems to increase performance. It might be because we want to load A to registers so it's better if there's less cluster sync there, but B is good for syncing. Either way, I could try the cluster plan
- I can also try doing more precision stuff by occasionally accumulating into fp32 from bf16 instead of everytime. I'll try that first before doing the cluster idea LOL