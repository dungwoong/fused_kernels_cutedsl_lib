# Planning this fusion

If we had a persistent kernel, you'd have a tile scheduler and it would give you a work tile idx that applies to stage 1 and 2.

There would be separate functions that handle each stage though. Where would we put warp spec if else?

Maybe we reinterpret SMEM at the start and pass it in. Then, our code structure will be like

```python
SMEM1, SMEM2 = ...
if producer:
    while work_tile:
        first_part(SMEM1)
        second_part(SMEM2)
        producer_tail, etc.
if consumer:
    while work_tile:
        first_part(SMEM1)
        ...
```

- In the `tile_scheduler` you have to pass in something like `tile_coord_mnkl = (head, head, None, head)` so you're using runtime values. This is because MLIR will extract runtime vs compile time values so this is just a bit of a hack to use their tile coord.
- be aware of what needs to be transposed since CuteDSL kernels compute AB.t() so e.g. X @ WQ, you must do WQ.t(). Check `dec_attn_run.py` for more details.

# More stuff

I THINK in triton you have to specifically pass in a descriptor to use TMA and you have to call warp specialize to warp specialize so the Trinity kernel has neither of these.

If we can, maybe 1 warp does the gemms, another warp does the softmax since low tile requirements.

OR 2 warps doing gemms with lower registers(e.g. 160) and another warpgroup doing softmax.

Accumulating stuff
- We can have each thread just hold its own stuff for the sum, until the end when we bfly reduce across 8 threads, and then across warps in the WG for 16 total items so then we can div by them
- this is the registers btw `((2,2,2),1,1):((1,2,4),0,0)` the first 2 is the 2 items, the second is 16x8(but transpose so 8x16), 3, 4 are the next row. We should get 4 values as output, and we have to sum along the second mode only
- The output registers will be like (2, 2)

# Debugging col sum
- I verified that starting from thread values to warp values, I should be correct by putting [1, 2, 3, 4] in the initial accumulator and tracking where data went. It's good UNLESS threads are messing up store locations
- If we set the acc to 1 instead of 0 initially, the thread accs should be 1 at the end, the warp accs will be 8, and the block accs will be 64. I test and the reg accs differ by 1, warp accs differ by 8 but the block accs have varying differences so the block acc must be the problem.

# Some bugs I had
- Make sure you keep track of what the layouts will be so you index in the right places
- I had things wrong so I was always indexing my one thing at 0 and indexing another thing on the moving thing and it was not good.