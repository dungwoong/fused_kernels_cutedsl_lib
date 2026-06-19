#define NUM_GPUS 1
#define USE_NVSHMEM false
#define MIRAGE_GRACE_HOPPER
#include "runtime.h"
using namespace cute;


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
    // OP type: kn_matmul_op
    half_t *dtensor10224949 = (half_t*)input_tensors.at(0);
    half_t *dtensor10224950 = (half_t*)input_tensors.at(1);
    half_t *dtensor10224951 = (half_t*)((char*)buf + 0);
    kn::gemm<CUBLAS_COMPUTE_16F>(dtensor10224951,dtensor10224949,dtensor10224950, 4096,4096,4096, 4096,1, 4096,1, 4096,1, 1, 0,0,0);
  }
  {
    // OP type: kn_relu_op
    half_t *dtensor10224951 = (half_t*)((char*)buf + 0);
    half_t *dtensor10224952 = (half_t*)output_tensors.at(0);
    using kernel = kn::ElementUnaryKernel<half_t, kn::ElementUnaryOpType::RELU, Layout<Shape<Int<4096>, Int<4096>>, Stride<Int<1>, Int<4096>>>, Layout<Shape<Int<4096>, Int<4096>>, Stride<Int<1>, Int<4096>>>>;
    kernel::run(dtensor10224952, dtensor10224951);
  }
  {
    // OP type: kn_output_op
  }
}