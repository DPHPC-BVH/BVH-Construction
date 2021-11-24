#include "CudaBVHBuilder.cuh"
#include "Common.h"

NAMESPACE_DPHPC_BEGIN

/**
 * Generate morton codes on GPU
 */
void GenerateMortonCodes32(int nPrimitives, unsigned int* dMortonCodes,
        BVHPrimitiveInfoWithIndex* dPrimitiveInfo)
{
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((nPrimitives + (blockSize.x - 1)) / blockSize.x, 1, 1);

    GenerateMortonCodes32Kernel<<<gridSize, blockSize>>>(nPrimitives, dMortonCodes, dPrimitiveInfo);
}


/**
 * Kernel to genrate morton codes with 32 bits
 */
__global__ void GenerateMortonCodes32Kernel(int nPrimitives, unsigned int* mortonCodes,
        BVHPrimitiveInfoWithIndex* primitiveInfo) {

    const int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= nPrimitives)
    {
        return;
    }

    mortonCodes[index] = GetMortonCode32(
        primitiveInfo[index].centroid.x,
        primitiveInfo[index].centroid.y,
        primitiveInfo[index].centroid.z);
    
}


NAMESPACE_DPHPC_END


