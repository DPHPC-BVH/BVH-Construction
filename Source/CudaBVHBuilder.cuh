#pragma once

#include "CudaBVHBuilder.h"


NAMESPACE_DPHPC_BEGIN

// Given the primitive info, construct all interior nodes
__global__ void ParallelConstructInteriorNodes(int nPrimitives, BVHPrimitiveInfoWithIndex* primitiveInfo_device, BVHBuildNodeDevice* interiorNodes_device);


NAMESPACE_DPHPC_END
