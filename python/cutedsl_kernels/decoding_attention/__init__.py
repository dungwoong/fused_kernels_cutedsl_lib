# this did the initial x * [qkv] matmul
# from .attempt1 import Kernel as DAttn1
# from .attempt2 import Kernel as DAttn2
from .attn1 import Kernel as DAttn1
from .attn2 import Kernel as DAttn2
from .attn3 import KernelSplitKGMEM as DAttnSplit1, ReduceDowncastKernel as AttnReduce1