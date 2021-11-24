#include "CudaBVHBuilder.cuh"

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

/**
 * Builds Tree hierarchy on GPU
 */
void BuildTreeHierarchy(int nPrimitives, unsigned int* dMortonCodesSorted,
        unsigned int* dIndicesSorted, CudaBVHBuildNode* dTree)
{
    dim3 blockSize(256, 1, 1);
    dim3 gridSize(((nPrimitives - 1) + (blockSize.x - 1)) / blockSize.x, 1, 1);

    BuildTreeHierarchyKernel<<<gridSize, blockSize>>>(nPrimitives, dMortonCodesSorted, dIndicesSorted, dTree);
}

__global__ void BuildTreeHierarchyKernel(int nPrimitives, unsigned int* mortonCodesSorted,
        unsigned int* indicesSorted, CudaBVHBuildNode* tree)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Do nothing if we have more threads than required
    if (i >= (nPrimitives - 1))
    {
        return;
    }

    // Thread i takes care of internal node with key 'key1'
    const unsigned int key1 = indicesSorted[i];

    // Determine direction of the range (+1 or -1)
    int lcp1 = LongestCommonPrefix(mortonCodesSorted, nPrimitives, i, i + 1, key1);
    int lcp2 = LongestCommonPrefix(mortonCodesSorted, nPrimitives, i, i - 1, key1);
    int d = sgn(lcp1 - lcp2);

    // Compute upper bound for the length of the range
    int minLcp = LongestCommonPrefix(mortonCodesSorted, nPrimitives, i, i - d, key1);
    int lMax = 2;
    while(LongestCommonPrefix(mortonCodesSorted, nPrimitives, i, i + lMax * d, key1) > minLcp) {
        lMax = lMax * 2;
    }

    // Find the other end using binary search
    int l = 0;
    int t = lMax / 2;
    while (t >= 1) {
        if(LongestCommonPrefix(mortonCodesSorted, nPrimitives, i, i + (l + t) * d, key1) > minLcp) {
            l += t;
        }
        t = t / 2;
    }
    int j = i + l * d;

    // Find the split position using binary search
    int nodeLcp = LongestCommonPrefix(mortonCodesSorted, nPrimitives, i, j, key1);
    int s = 0;
    t = divCeil(l,2);
    while (t >= 1) {
        if(LongestCommonPrefix(mortonCodesSorted, nPrimitives, i, i + (s + t) * d, key1) > nodeLcp) {
            s += t;
        }
        t = divCeil(t,2);
    }
    int splitPosition = i + s * d + min(d, 0);

    // Update child indices
    int leftIndex;
    int rightIndex;

    // Case: Left child
    if (min(i, j) == splitPosition) {
        // child is a leaf
        leftIndex = splitPosition + nPrimitives - 1;

        tree[leftIndex].children[0] = -1;
        tree[leftIndex].children[1] = -1;
        tree[leftIndex].parent = splitPosition;

        tree[leftIndex].dataIdx = indicesSorted[splitPosition];
        
    } else {
        // child is interior node
        leftIndex = splitPosition;
    }

    // Case: Right child
    if (max(i, j) == splitPosition + 1) {
        // child is a leaf
        rightIndex = splitPosition + 1 + nPrimitives - 1;

        tree[rightIndex].children[0] = -1;
        tree[rightIndex].children[1] = -1;
        tree[rightIndex].parent = splitPosition;

        tree[rightIndex].dataIdx = indicesSorted[splitPosition + 1];
        
    } else {
        // child is interior node
        rightIndex = splitPosition + 1;
    }

    tree[i].children[0] = leftIndex;
    tree[i].children[1] = rightIndex;

    // Update parent index of children
    tree[leftIndex].parent = i;
    tree[rightIndex].parent = i;

    // Handle special case of the root
    if(i == 0) {
        tree[0].parent = -1;
    } 
}

__device__ int LongestCommonPrefix(unsigned int* sortedKeys, unsigned int numberOfElements,
        int index1, int index2, unsigned int key1)
{
    // No need to check the upper bound, since i+1 will be at most numberOfElements - 1 (one 
    // thread per internal node)
    if (index2 < 0 || index2 >= numberOfElements)
    {
        return 0;
    }

    unsigned int key2 = sortedKeys[index2];

    // Fallback if k_i = k_j, identic keys have to be handled carefully
    if (key1 == key2)
    {
        return 32 + __clz(index1 ^ index2);
    }

    return __clz(key1 ^ key2);
}



NAMESPACE_DPHPC_END


