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

# More stuff

I THINK in triton you have to specifically pass in a descriptor to use TMA and you have to call warp specialize to warp specialize so the Trinity kernel has neither of these.