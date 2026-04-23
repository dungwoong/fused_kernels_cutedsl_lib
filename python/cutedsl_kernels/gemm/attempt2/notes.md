# Try to dispatch next MMAs before doing epilogue

- the producer will keep producing
- the consumer will finish mainloop, then advance worktile, then dispatch next MMAs, and then do the epilogue, and then yeah

- first step: I want a utility function that can run gemm mainloop. Make sure no performance degradation. THEN, move onto using this for transition GEMM