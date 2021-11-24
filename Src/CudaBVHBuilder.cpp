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
	BVHPrimitiveInfoWithIndex* dPrimitiveInfo;
	cudaMalloc(&dPrimitiveInfo, sizeof(BVHPrimitiveInfoWithIndex) * primitiveInfo.size());
	cudaMemcpy(dPrimitiveInfo, primitiveInfo.data(), sizeof(BVHPrimitiveInfoWithIndex) * primitiveInfo.size(), cudaMemcpyHostToDevice);

	// 1. Compute Morton Codes
	unsigned int* dMortonCodes;
	cudaMalloc(&dMortonCodes, sizeof(unsigned int*) * primitiveInfo.size());
	unsigned int* dMortonIndices;
	cudaMalloc(&dMortonCodes, sizeof(unsigned int*) * primitiveInfo.size());
	GenerateMortonCodes32(primitiveInfo.size(), dPrimitiveInfo, dMortonCodes, dMortonIndices);
	
	// 2. Sort Morton Codes
	unsigned int* dMortonCodesSorted;
	cudaMalloc(&dMortonCodes, sizeof(unsigned int*) * primitiveInfo.size());
	unsigned int* dMortonIndicesSorted;
	cudaMalloc(&dMortonCodes, sizeof(unsigned int*) * primitiveInfo.size());
	DeviceSort(primitiveInfo.size(), &dMortonCodes, &dMortonCodesSorted,
                 &dMortonIndices, &dMortonIndicesSorted);

	
	// 3. Build tree hierarchy of CudaBVHBuildNodes

	
	// 4. Compute Bounding Boxes of each node

	
	// 5. Flatten Tree


	// 6. Don't forget to free  memory
	cudaFree(dPrimitiveInfo);


}


NAMESPACE_DPHPC_END


