# Adding tile layout (1, 2) instead of (2, 1)
- If you have (1, 2) your epilogue tile has to be 64x(cta_tile_n) so all warps can store on each step, and you iterate down the epilogue tile. Otherwise, we'd have to modify the epilogue so warps take turns storing or whatever
- After mathing out the stmatrix, I found when we populate mma we do 64x(cta_tile_n), giving that mma size to EACH mma, which is not what we want.
- so when make trivial tiled mma, do `tiler_mn=(64, self.cta_tile_shape_mnk[1] // self.atom_layout_mnk[1])`
- fixed!

# Try to dispatch next MMAs before doing epilogue

- the producer will keep producing
- the consumer will finish mainloop, then advance worktile, then dispatch next MMAs, and then do the epilogue, and then yeah

- first step: I want a utility function that can run gemm mainloop. Make sure no performance degradation. THEN, move onto using this for transition GEMM

Ok so if you do it before the epilogue happens(e.g. before stmatrix) then obviously it will spill. It will allocate a totally new set of registers for the new accumulators. HOWEVER, what if we try it after stmatrix, before the tma store so you just quickly pause to stmatrix and then you just go? I think I can also get the epilogue size up since I'm only using 3 stages.

# Horizontal tiling

```
Tiled MMA
  Thr Layout VMNK: (128,1,2,1):(1,0,128,0)
  Permutation MNK: (_,(8,32,2):(1,16,8),_)
MMA Atom
  ThrID:           128:1
  Shape MNK:       (64,256,16)
  TV Layout A:     (128,(64,16)):(0,(1,64))
  TV Layout B:     (128,(256,16)):(0,(1,256))
  TV Layout C:     ((4,8,4),(2,2,32)):((128,1,16),(64,8,512))
```

So note the CTAs get data in colexicographical order so the first CTA will get 8 elements, then skip 8, then 8 more etc.

- let's say K is 32, (512+64)x32=18432, (128+256)x32=12288. Same output tile size, but different arithmetic intensity.
- Reminder that the epilogue is still going left to right since WGs are interleaved

Some sizes that work for m=64, n=512
- k=64, ab_stage=3, epi_tile_mn=(64, 32), epi_stage=2. Can't increase a single parameter here btw
- k=32, ab_stage=5, epi_tile_mn=(64, 64), epi_stage=2
- both of these get you ~90% performance. I think the k=32 is slightly better

- on mnk 2048 8192 4096 you can get up to 93% performance
- on mnk 4096 4096 4096 you can get like 90%
- on that BS 8192 8192 64 shape you get 1.02 so thank god
- on something slightly more realistic like 1024 16384 4096 (up cast) you can get like 0.93

Looking at the NCU report
- we see for e.g. 4096 we had compute/mem throughput of 90 and 70 but horizontal you get 89 and 81 so I guess since there was room for whatever you are able to do stuff.
- the issue could potentially be since we have to write back slowly, the throughput is lower. But honestly idek because the epi tile is 64, 64 and we used to have 128 32
- I could try increasing gemm n prologue or smth. Ok that did not work. 1 is good. If 1 is 92%, 0 is 90% so it's ok to be at 0 which is good for rmsnorm + linear or auto-codegen
- We could try better rasterization with the scheduler to get a better hitrate on the B matrix potentially but I'm not sure