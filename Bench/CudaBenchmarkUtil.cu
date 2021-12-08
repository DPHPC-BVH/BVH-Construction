#include "CudaBenchmarkUtil.cuh"


NAMESPACE_DPHPC_BEGIN

/**
 * Warms up GPU for benchmarking
 */
void WarmUpGPU()
{
    dim3 blockSize(256, 1, 1);
    dim3 gridSize(1024, 1, 1);

    WarmUpGPUKernel<<<gridSize, blockSize>>>();
}

__global__ void WarmUpGPUKernel(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}



NAMESPACE_DPHPC_END
