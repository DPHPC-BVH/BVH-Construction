#pragma once

#include "CudaBVHBuilder.h"
#include <cuda.h>
#include <cuda_runtime.h>


NAMESPACE_DPHPC_BEGIN

// Given the primitive info, construct all interior nodes
__global__ void ParallelConstructInteriorNodes(int nPrimitives, BVHPrimitiveInfoWithIndex* primitiveInfo_device, BVHBuildNodeDevice* interiorNodes_device);

__global__ void GenerateMortonCodes32(int nPrimitives, unsigned int* mortonCodes, BVHPrimitiveInfoWithIndex* primitiveInfo_device);

__forceinline__ __device__ uint32_t LeftShiftAndExpand32(uint32_t x)
{
    if (x == (1 << 10)) --x; // Why?
    x = (x | (x << 16)) & 0x30000ff;
    // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x | (x << 8)) & 0x300f00f;
    // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x | (x << 4)) & 0x30c30c3;
    // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x | (x << 2)) & 0x9249249;
    // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}

/**
 * Function that computes the 32 bit morton code of a 3D-Point where x,y,z in [0,1]
 */
__forceinline__ __device__ uint32_t getMortonCode32(float x, float y, float z) {
    return (LeftShiftAndExpand32(1024 * z) << 2) | (LeftShiftAndExpand32(1024 * y) << 1) | LeftShiftAndExpand32(1024 * x);
}

NAMESPACE_DPHPC_END
