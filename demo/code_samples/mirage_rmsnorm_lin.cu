#define NUM_GPUS 1
#define USE_NVSHMEM false
#define MIRAGE_GRACE_HOPPER
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000114_ptr, half_t const* __restrict__ dtensor10000112_ptr, half_t const* __restrict__ dtensor10000113_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000599_ptr = (half_t*)(buf + 128);
  half_t *stensor20000596_ptr = (half_t*)(buf + 1568);
  half_t *stensor20000598_ptr = (half_t*)(buf + 1152);
  half_t *stensor30000591_ptr = (half_t*)(buf + 14976);
  half_t *stensor20000591_ptr = (half_t*)(buf + 1664);
  half_t *stensor30000590_ptr = (half_t*)(buf + 1408);
  half_t *stensor20000590_ptr = (half_t*)(buf + 1152);
  half_t *stensor20000594_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000112 -> stensor 20000590
  const half_t *dtensor10000112_tile_ptr = dtensor10000112_ptr ;
  using DTensor10000112TileLayout = Layout<Shape<Int<64>, Int<2>>, Stride<Int<1>, Int<4096>>>;
  using STensor20000590InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<2>>, Stride<Int<1>, Int<64>>>{})), DTensor10000112TileLayout, NUM_THREADS>;
  half_t *stensor20000590_async_copy_buf = stensor30000590_ptr;
  // Copy for G->S: dtensor 10000113 -> stensor 20000591
  const half_t *dtensor10000113_tile_ptr = dtensor10000113_ptr  + blockIdx.x*96*1;
  using DTensor10000113TileLayout = Layout<Shape<Int<96>, Int<64>>, Stride<Int<1>, Int<6144>>>;
  using STensor20000591InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<96>, Int<64>>, Stride<Int<1>, Int<104>>>, DTensor10000113TileLayout, NUM_THREADS>;
  half_t *stensor20000591_async_copy_buf = stensor30000591_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000599 -> dtensor 10000114
  half_t *dtensor10000114_tile_ptr = dtensor10000114_ptr  + blockIdx.x*96*1;
  using DTensor10000114TileLayout = Layout<Shape<Int<96>, Int<2>>, Stride<Int<1>, Int<6144>>>;
  using STensor20000599OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000114TileLayout, Layout<Shape<Int<96>, Int<2>>, Stride<Int<1>, Int<96>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 512, NUM_THREADS>::run(stensor20000594_ptr, thread_idx);
  
  
  using Matmul20000598LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<2>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000598LayoutB = Layout<Shape<Int<96>, Int<64>>, Stride<Int<1>, Int<104>>>;
  using Matmul20000598LayoutC = Layout<Shape<Int<96>, Int<2>>, Stride<Int<1>, Int<104>>>;
  using Matmul20000598LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000598LayoutBAligned = Layout<Shape<Int<96>, Int<64>>, Stride<Int<1>, Int<96>>>;
  using Matmul20000598Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000598LayoutA, Matmul20000598LayoutB, Matmul20000598LayoutC, Matmul20000598LayoutAAligned, Matmul20000598LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000598_accum = Matmul20000598Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20000591InputAtom::run(stensor20000591_async_copy_buf, dtensor10000113_tile_ptr, thread_idx);
    STensor20000590InputAtom::run(stensor20000590_async_copy_buf, dtensor10000112_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 64; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 64) {
        STensor20000591InputAtom::run(stensor20000591_ptr, dtensor10000113_tile_ptr + 393216*(for_idx+1), thread_idx);
        STensor20000590InputAtom::run(stensor20000590_ptr, dtensor10000112_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000591_ptr, stensor20000591_async_copy_buf);
      SWAP(stensor20000590_ptr, stensor20000590_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_square_op
      using InLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<2>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
      using OutLayout = Layout<Shape<Int<2>, Int<64>>, Stride<Int<1>, Int<8>>>;
      using Kernel = tb::ElementUnaryKernel<half_t, tb::ElementUnaryOpType::SQUARE, OutLayout, InLayout, NUM_THREADS, tb::EpilogueMulScalar<half_t, tb::EpilogueStoreAccum<half_t>>>;
      const float scalars[] = {0.000244f, 0.0f};
      Kernel::run(stensor20000594_ptr, stensor20000590_ptr, thread_idx, 0.000000, scalars);
    }
    {
      // OP type: tb_matmul_op
      Matmul20000598Kernel::run(matmul_20000598_accum, stensor20000590_ptr, stensor20000591_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000598Kernel::write_back_mma_rC(stensor20000598_ptr, matmul_20000598_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_1_op
    using InLayout = Layout<Shape<Int<2>, Int<64>>, Stride<Int<1>, Int<8>>>;
    using OutLayout = Layout<Shape<Int<2>, Int<1>>, Stride<Int<1>, Int<8>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueSqrt<half_t, tb::EpilogueStore<half_t>>>;
    const float scalars[] = {0.000000f, 0.0f};
    Kernel::run(stensor20000596_ptr, stensor20000594_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<96>, Int<2>>, Stride<Int<1>, Int<104>>>;
    using In1Layout = Layout<Shape<Int<1>, Int<2>>, Stride<Int<8>, Int<1>>>;
    using OutLayout = Layout<Shape<Int<96>, Int<2>>, Stride<Int<1>, Int<96>>>;
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000599_ptr, stensor20000598_ptr, stensor20000596_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000599OutputAtom::run(dtensor10000114_tile_ptr, stensor20000599_ptr, thread_idx);
  }
}


static void _init() {
}


static void _execute_mugraph(std::vector<void const *> input_tensors, std::vector<void*> output_tensors, void* buf, cudaStream_t stream, void * profiler_buffer){
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000114 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000112 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000113 = (half_t*)input_tensors.at(1);
    dim3 grid_dim(64, 1, 1);
    dim3 block_dim(256, 1, 1);
    size_t smem_size = 28288;
    
    // define tmas
    std::vector<bool> minputs = {};
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 28288);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000114, dtensor10000112, dtensor10000113);
  }
  {
    // OP type: kn_output_op
  }
}