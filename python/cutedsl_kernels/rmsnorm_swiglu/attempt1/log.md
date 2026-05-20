# First try
- copied RMS + Linear code
- just need to add the second GEMM and modify the epilogue
- only use 1 pipeline for everything, that's what the other swiglu does

- since you do the silu, you have to scale the stuff separately cuz it doesn't distribute thru