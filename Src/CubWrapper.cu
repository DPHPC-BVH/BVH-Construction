#include "CubWrapper.cuh"
#include <cub/cub.cuh>


NAMESPACE_DPHPC_BEGIN

template <typename T> void DeviceSort(unsigned int numberOfElements, T** dKeysIn, T** dKeysOut,
                 unsigned int** dValuesIn, unsigned int** dValuesOut)
{   

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<T> dKeys(*dKeysIn, *dKeysOut);
    cub::DoubleBuffer<unsigned int> dValues(*dValuesIn, *dValuesOut);
    
    // Determine temporary device storage requirements
    void     *dTempStorage = NULL;
    size_t   dTempStorageBytes = 0;
    cub::DeviceRadixSort::SortPairs(dTempStorage, dTempStorageBytes, dKeys, dValues, numberOfElements);
    // Allocate temporary storage
    cudaMalloc(&dTempStorage, dTempStorageBytes);
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(dTempStorage, dTempStorageBytes, dKeys, dValues, numberOfElements);
    // Free temporary memory
    cudaFree(dTempStorage);
    // Update out buffers
    T* current = dKeys.Current();
    dKeysOut = &current;
    unsigned int* current2 = dValues.Current();
    dValuesOut = &current2;

}

void DeviceSort(unsigned int numberOfElements, unsigned int** dKeysIn, unsigned int** dKeysOut,
                 unsigned int** dValuesIn, unsigned int** dValuesOut) {
    DeviceSort<unsigned int>(numberOfElements, dKeysIn, dKeysOut, dValuesIn, dValuesOut);
}

NAMESPACE_DPHPC_END
