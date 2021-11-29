#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "CubWrapper.cuh"


NAMESPACE_DPHPC_BEGIN

TEST(CubWrapperTest, DeviceSort) {
  
  const int numItems = 7;
  unsigned int keysIn[7] = {8, 6, 7, 5, 3, 0, 9};
  unsigned int valuesIn[7] = {0, 1, 2, 3, 4, 5, 6};
  unsigned int keysOut[7];
  unsigned int valuesOut[7];

  
   // Allocate memory 
  unsigned int  *dKeysIn;
  unsigned int  *dKeysOut;
  unsigned int  *dValuesIn;
  unsigned int  *dValuesOut;

  cudaMalloc(&dKeysIn, numItems * sizeof(unsigned int));
  cudaMalloc(&dKeysOut, numItems * sizeof(unsigned int));
  cudaMalloc(&dValuesIn, numItems * sizeof(unsigned int));
  cudaMalloc(&dValuesOut, numItems * sizeof(unsigned int));

  // Copy input to Device
  cudaMemcpy(dKeysIn, keysIn, sizeof(unsigned int) * numItems, cudaMemcpyHostToDevice);
  cudaMemcpy(dValuesIn, valuesIn, sizeof(unsigned int) * numItems, cudaMemcpyHostToDevice);

  // Perform sort on Device
  DeviceSort(numItems, &dKeysIn, &dKeysOut, &dValuesIn, &dValuesOut);

  // Copy results from Device to Host
  cudaMemcpy(keysOut, dKeysOut, sizeof(unsigned int) * numItems, cudaMemcpyDeviceToHost);
  cudaMemcpy(valuesOut, dValuesOut, sizeof(unsigned int) * numItems, cudaMemcpyDeviceToHost);

  // Free memory
  cudaFree(dKeysIn);
  cudaFree(dKeysOut);
  cudaFree(dValuesIn);
  cudaFree(dValuesOut);

  
  unsigned int keysOutExpected[7] = {0, 3, 5, 6, 7, 8, 9};
  for (size_t i = 0; i < numItems; i++)
    EXPECT_EQ(keysOut[i], keysOutExpected[i]);
  
  unsigned int valuesOutExpected[7] = {5, 4, 3, 1, 2, 0, 6};
  for (size_t i = 0; i < numItems; i++)
    EXPECT_EQ(valuesOut[i], valuesOutExpected[i]);
  
}


NAMESPACE_DPHPC_END