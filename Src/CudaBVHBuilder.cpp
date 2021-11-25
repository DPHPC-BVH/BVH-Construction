#include "CudaBVHBuilder.h"
#include "CudaBVHBuilder.cuh"
#include "CubWrapper.cuh"

NAMESPACE_DPHPC_BEGIN



CudaBVHBuilder::CudaBVHBuilder(BVH& bvh) 
	: BVHBuilder(bvh)
{
	// We only need to know each bounding box of the primitive to construct the BVH.
	primitiveInfo.reserve(bvh.primitives.size());
	for (int i = 0; i < bvh.primitives.size(); ++i) {
		primitiveInfo.push_back({ i, bvh.primitives[i]->WorldBound() });
	}
}


CudaBVHBuilder::~CudaBVHBuilder() {
}


void CudaBVHBuilder::BuildBVH() {

	// For BVH construction we only need the bounding boxes and the centroids of the primitive
	const unsigned int nPrimitives = primitiveInfo.size();

	BVHPrimitiveInfoWithIndex* dPrimitiveInfo;
	cudaMalloc(&dPrimitiveInfo, sizeof(BVHPrimitiveInfoWithIndex) * nPrimitives);
	cudaMemcpy(dPrimitiveInfo, primitiveInfo.data(), sizeof(BVHPrimitiveInfoWithIndex) * nPrimitives, cudaMemcpyHostToDevice);

	// 1. Compute Morton Codes
	unsigned int* dMortonCodes;
	cudaMalloc(&dMortonCodes, sizeof(unsigned int) * nPrimitives);
	unsigned int* dMortonIndices;
	cudaMalloc(&dMortonIndices, sizeof(unsigned int) * nPrimitives);
	GenerateMortonCodes32(nPrimitives, dPrimitiveInfo, dMortonCodes, dMortonIndices);
	
	// 2. Sort Morton Codes
	unsigned int* dMortonCodesSorted;
	cudaMalloc(&dMortonCodesSorted, sizeof(unsigned int) * nPrimitives);
	unsigned int* dMortonIndicesSorted;
	cudaMalloc(&dMortonIndicesSorted, sizeof(unsigned int) * nPrimitives);
	DeviceSort(nPrimitives, &dMortonCodes, &dMortonCodesSorted,
                 &dMortonIndices, &dMortonIndicesSorted);
	cudaFree(dMortonCodes);
	cudaFree(dMortonIndices);

	// 3. Build tree hierarchy of CudaBVHBuildNodes
	CudaBVHBuildNode* dTree;
	cudaMalloc(&dTree, sizeof(CudaBVHBuildNode) * (2 * nPrimitives - 1));
	BuildTreeHierarchy(nPrimitives, dMortonCodesSorted, dMortonIndicesSorted, dTree);
	cudaFree(dMortonCodesSorted);
	cudaFree(dMortonIndicesSorted);

	// 4. Compute Bounding Boxes of each node
	ComputeBoundingBoxes(nPrimitives, dTree, dPrimitiveInfo);
	CudaBVHBuildNode treeWithBoundingBoxes[2*nPrimitives - 1];
  	cudaMemcpy(treeWithBoundingBoxes, dTree, (2 * nPrimitives - 1) * sizeof(CudaBVHBuildNode), cudaMemcpyDeviceToHost);
	cudaFree(dPrimitiveInfo);
	cudaFree(dTree);

	// 5. Flatten Tree


}


NAMESPACE_DPHPC_END


