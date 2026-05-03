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