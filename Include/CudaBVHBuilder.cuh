#pragma once

#include "CudaBVHBuilder.h"
#include <cuda.h>
#include <cuda_runtime.h>

NAMESPACE_DPHPC_BEGIN

void GenerateMortonCodes32(int nPrimitives, BVHPrimitiveInfoWithIndex* dPrimitiveInfo,
        unsigned int* dMortonCodes, unsigned int* dIndices);

__global__ void GenerateMortonCodes32Kernel(int nPrimitives, BVHPrimitiveInfoWithIndex* primitiveInfo,
        unsigned int* dMortonCodes, unsigned int* indices);


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
__forceinline__ __device__ uint32_t GetMortonCode32(float x, float y, float z) {
    return (LeftShiftAndExpand32(1024 * z) << 2) | (LeftShiftAndExpand32(1024 * y) << 1) | LeftShiftAndExpand32(1024 * x);
}

void BuildTreeHierarchy(int nPrimitives, unsigned int* dMortonCodesSorted,
        unsigned int* dIndicesSorted, CudaBVHBuildNode* dTree);

__global__ void BuildTreeHierarchyKernel(int nPrimitives, unsigned int* mortonCodesSorted,
        unsigned int* indicesSorted, CudaBVHBuildNode* tree);

__device__ int LongestCommonPrefix(unsigned int* sortedKeys, unsigned int numberOfElements,
        int index1, int index2, unsigned int key1);

void ComputeBoundingBoxes(int nPrimitives, CudaBVHBuildNode* tree, BVHPrimitiveInfoWithIndex* primitiveInfo);

__global__ void ComputeBoundingBoxesKernel(int nPrimitives, CudaBVHBuildNode* tree, BVHPrimitiveInfoWithIndex* primitiveInfo, int* interiorNodeCounter);

__device__ void BoundingBoxUnion(Bounds3f bIn1, Bounds3f bIn2, Bounds3f* bOut);


/**
 * Computes the sign function
 */
template <typename T> __forceinline__ __device__ __host__ int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}
/**
 * Computes ceil(x / y) for x >= 0 and y > 0
 */
__forceinline__ __device__ __host__ int divCeil(int x, int y) {
    return x / y + (x % y != 0);
}

NAMESPACE_DPHPC_END
