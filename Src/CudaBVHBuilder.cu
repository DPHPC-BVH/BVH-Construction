#include "CudaBVHBuilder.cuh"

NAMESPACE_DPHPC_BEGIN



/**
 * Generate morton codes on GPU
 */
void GenerateMortonCodes32(int nPrimitives, int stride, BVHPrimitiveInfoWithIndex* dPrimitiveInfo,
        unsigned int* dMortonCodes, unsigned int* dIndices)
{
    int nTasksPerBlock = blockSize * stride;
    int gridSize = (nPrimitives + (nTasksPerBlock - 1)) / blockSize;
    GenerateMortonCodes32Kernel<<<gridSize, blockSize>>>(nPrimitives, stride, dPrimitiveInfo, dMortonCodes, dIndices);
}


/**
 * Kernel to genrate morton codes with 32 bits
 */
__global__ void GenerateMortonCodes32Kernel(int nPrimitives, int stride, BVHPrimitiveInfoWithIndex* primitiveInfo,
        unsigned int* mortonCodes, unsigned int* indices) {

    const int gIdx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = gIdx * stride; i < (gIdx + 1) * stride; ++i) {
        if (i >= nPrimitives) {
            return;
        }

        mortonCodes[i] = GetMortonCode32(
            primitiveInfo[i].centroid.x,
            primitiveInfo[i].centroid.y,
            primitiveInfo[i].centroid.z);

        indices[i] = i;
    }
}

/**
 * Builds Tree hierarchy on GPU
 */
void BuildTreeHierarchy(int nPrimitives, int stride, unsigned int* dMortonCodesSorted,
        unsigned int* dIndicesSorted, CudaBVHBuildNode* dTree)
{
    int nTasksPerBlock = blockSize * stride;
    int gridSize = (nPrimitives-1 + (nTasksPerBlock - 1)) / blockSize;
    BuildTreeHierarchyKernel<<<gridSize, blockSize>>>(nPrimitives, stride, dMortonCodesSorted, dIndicesSorted, dTree);
}

__global__ void BuildTreeHierarchyKernel(int nPrimitives, int stride, unsigned int* mortonCodesSorted,
        unsigned int* indicesSorted, CudaBVHBuildNode* tree)
{
    const int gIdx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = gIdx * stride; i < (gIdx + 1) * stride; ++i) {

        // Do nothing if we have more threads than required
        if (i >= (nPrimitives - 1)) {
            return;
        }

        // Thread i takes care of internal node with key 'key1'
        const unsigned int key1 = mortonCodesSorted[i];

        // Determine direction of the range (+1 or -1)
        int lcp1 = LongestCommonPrefix(mortonCodesSorted, nPrimitives, i, i + 1, key1);
        int lcp2 = LongestCommonPrefix(mortonCodesSorted, nPrimitives, i, i - 1, key1);
        int d = sgn(lcp1 - lcp2);

        // Compute upper bound for the length of the range
        int minLcp = LongestCommonPrefix(mortonCodesSorted, nPrimitives, i, i - d, key1);
        int lMax = 2;
        while (LongestCommonPrefix(mortonCodesSorted, nPrimitives, i, i + lMax * d, key1) > minLcp) {
            lMax = lMax * 2;
        }

        // Find the other end using binary search
        int l = 0;
        int t = lMax;
        while (t > 1) {
            t = t / 2;
            if (LongestCommonPrefix(mortonCodesSorted, nPrimitives, i, i + (l + t) * d, key1) > minLcp) {
                l += t;
            }
        }
        int j = i + l * d;

        // Find the split position using binary search
        int nodeLcp = LongestCommonPrefix(mortonCodesSorted, nPrimitives, i, j, key1);
        int s = 0;
        t = l;
        int divisor = 2;
        const int maxDivisor = 1 << (32 - __clz(l));
        while (divisor <= maxDivisor) {
            t = divCeil(l, divisor);
            if (LongestCommonPrefix(mortonCodesSorted, nPrimitives, i, i + (s + t) * d, key1) > nodeLcp) {
                s += t;
            }
            divisor *= 2;
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

        }
        else {
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

        }
        else {
            // child is interior node
            rightIndex = splitPosition + 1;
        }

        tree[i].children[0] = leftIndex;
        tree[i].children[1] = rightIndex;

        // Update parent index of children
        tree[leftIndex].parent = i;
        tree[rightIndex].parent = i;

        // Set dataIdx to -1, i is since interior node
        tree[i].dataIdx = -1;

        // Handle special case of the root
        if (i == 0) {
            tree[0].parent = -1;
        }
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

/**
 * Computes bounding boxes in Tree hierarchy on GPU
 */
void ComputeBoundingBoxes(int nPrimitives, int stride, CudaBVHBuildNode* dTree, BVHPrimitiveInfoWithIndex* dPrimitiveInfo) {
    
    int nTasksPerBlock = blockSize * stride;
    int gridSize = (nPrimitives + (nTasksPerBlock - 1)) / blockSize;

    int* dInteriorNodeCounter;
    cudaMalloc(&dInteriorNodeCounter, (nPrimitives - 1) * sizeof(int));
    cudaMemset(dInteriorNodeCounter, -1, (nPrimitives - 1) * sizeof(int));

    ComputeBoundingBoxesKernel<<<gridSize, blockSize>>>(nPrimitives, stride, dTree, dPrimitiveInfo, dInteriorNodeCounter);

    cudaFree(dInteriorNodeCounter);
}


/**
  * Kernel to compute the bounding boxes
  * 
  * Remakrs: Makes not use of shared memory. One could improve this implementation 
  * by using shared memory to cache the information to compute the bounding boxes
  * 
  */
__global__ void ComputeBoundingBoxesKernel(int nPrimitives, int stride, CudaBVHBuildNode* tree, BVHPrimitiveInfoWithIndex* primitiveInfo, int* interiorNodeCounter) {
   
    const int gIdx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = gIdx * stride; i < (gIdx + 1) * stride; ++i) {

        // Do nothing if we have more threads than required
        if (i >= nPrimitives) {
            return;
        }

        const int index = i + (nPrimitives - 1);

        tree[index].bounds = primitiveInfo[tree[index].dataIdx].bounds;

        int currentIndex = tree[index].parent;
        while (currentIndex != -1) {

            int lastVisitedThreadId = atomicExch(&interiorNodeCounter[currentIndex], i);
            if (lastVisitedThreadId == -1) {
                // We are the first thread to visit the interior node, so we return
                return;
            }

            int leftIndex = tree[currentIndex].children[0];
            int rightIndex = tree[currentIndex].children[1];

            BoundingBoxUnion(tree[leftIndex].bounds, tree[rightIndex].bounds, &tree[currentIndex].bounds);

            currentIndex = tree[currentIndex].parent;

        }
    }
}

__device__ void BoundingBoxUnion(Bounds3f bIn1, Bounds3f bIn2, Bounds3f* bOut) {

    bOut->pMin.x = min(bIn1.pMin.x, bIn2.pMin.x);
    bOut->pMin.y = min(bIn1.pMin.y, bIn2.pMin.y);
    bOut->pMin.z = min(bIn1.pMin.z, bIn2.pMin.z);

    bOut->pMax.x = max(bIn1.pMax.x, bIn2.pMax.x);
    bOut->pMax.y = max(bIn1.pMax.y, bIn2.pMax.y);
    bOut->pMax.z = max(bIn1.pMax.z, bIn2.pMax.z);

}


NAMESPACE_DPHPC_END


