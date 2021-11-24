#include "CudaBVHBuilder.cuh"
#include "Common.h"

NAMESPACE_DPHPC_BEGIN

/**
 * Generate morton codes on GPU
 */
void GenerateMortonCodes32(int nPrimitives, BVHPrimitiveInfoWithIndex* dPrimitiveInfo,
        unsigned int* dMortonCodes, unsigned int* dIndices)
{
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((nPrimitives + (blockSize.x - 1)) / blockSize.x, 1, 1);

    GenerateMortonCodes32Kernel<<<gridSize, blockSize>>>(nPrimitives, dPrimitiveInfo, dMortonCodes, dIndices);
}


/**
 * Kernel to genrate morton codes with 32 bits
 */
__global__ void GenerateMortonCodes32Kernel(int nPrimitives, BVHPrimitiveInfoWithIndex* primitiveInfo,
        unsigned int* mortonCodes, unsigned int* indices) {

    const int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= nPrimitives)
    {
        return;
    }

    mortonCodes[index] = GetMortonCode32(
        primitiveInfo[index].centroid.x,
        primitiveInfo[index].centroid.y,
        primitiveInfo[index].centroid.z);
    
    indices[index] = index;
    
}


NAMESPACE_DPHPC_END


