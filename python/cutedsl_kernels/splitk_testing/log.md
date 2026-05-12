- Strategy is to store out to partitions in GMEM(FP32) and then combine to match the precision of cuBLAS. Alternatively you could use tma reduce store to gain speed and lose precision.
- I get around a 5% speedup over cublas if I use a PyTorch compile kernel for my reduce.
- The store out format is (m, SPLITS, n) so combining splits and n you can vectorize copy in
- with TMA you must have the last stride as 1. Having SPLITS as the second dimension helps with the reduce kernel. 

### Reduce combine kernel
- in quack, they have massive #cols, so you can have each CTA do a single row reduction.
- Here, we can have a single thread do a row reduction, and decide how large the blocks will be
