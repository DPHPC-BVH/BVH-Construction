#include "CudaBenchmarkUtil.cuh"


NAMESPACE_DPHPC_BEGIN

float WarmUpGPU() {   

    int blocks = 1 << 10;
    dim3 blockSize(256, 1, 1);
    dim3 gridSize(blocks, 1, 1);

    int size = blocks * 256;
    float* data = (float*) malloc(size * sizeof(float));
    for (size_t i = 0; i < size; i++)
    {
      data[i] = rand();
    }

    float* dData;
    cudaMalloc(&dData, size * sizeof(float));
    cudaMemcpy(dData, data, size * sizeof(float), cudaMemcpyHostToDevice);
    
    WarmUpGPUKernel<<<gridSize, blockSize>>>(dData);
    
    cudaMemcpy(data, dData, size * sizeof(float), cudaMemcpyDeviceToHost);

    float checksum = 0.0;
    for (size_t i = 0; i < size; i++)
    {
      checksum += data[i];
    }
    return checksum;
}

__global__ void WarmUpGPUKernel(float* dResults) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  volatile float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
  dResults[tid] += ib;
}



NAMESPACE_DPHPC_END
