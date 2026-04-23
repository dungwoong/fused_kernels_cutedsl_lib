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