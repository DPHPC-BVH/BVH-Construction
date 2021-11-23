#include "CudaBVHBuilder.cuh"

NAMESPACE_DPHPC_BEGIN

__global__ void ParallelConstructInteriorNodes(int nPrimitives, BVHPrimitiveInfoWithIndex* primitiveInfo_device, BVHBuildNodeDevice* interiorNodes_device)
{
    // May need to initialize primitiveInfo_device first (its idx)
	
}

/**
 * Generates morton codes
 */
__global__ void GenerateMortonCodes32(int nPrimitives, unsigned int* mortonCodes, BVHPrimitiveInfoWithIndex* primitiveInfo_device) {

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i=index; i < nPrimitives; i+= stride) {
        mortonCodes[i] = getMortonCode32(
            primitiveInfo_device[i].centroid.x,
            primitiveInfo_device[i].centroid.y,
            primitiveInfo_device[i].centroid.z);
    }
}


NAMESPACE_DPHPC_END


