#include "CudaBVHBuilder.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

NAMESPACE_DPHPC_BEGIN

__global__ void ParallelConstructInteriorNodes(int nPrimitives, BVHPrimitiveInfoWithIndex* primitiveInfo_device, BVHBuildNodeDevice* interiorNodes_device)
{
    // May need to initialize primitiveInfo_device first (its idx)
	
}

NAMESPACE_DPHPC_END

