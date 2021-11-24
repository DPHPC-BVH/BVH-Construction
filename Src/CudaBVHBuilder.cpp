#include "CudaBVHBuilder.h"
#include "CudaBVHBuilder.cuh"

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
	
	
	// 2. Sort Morton Codes

	
	// 3. Build tree hierarchy of CudaBVHBuildNodes

	
	// 4. Compute Bounding Boxes of each node

	
	// 5. Flatten Tree


	// 6. Don't forget to free  memory
	cudaFree(dPrimitiveInfo);


}


NAMESPACE_DPHPC_END


