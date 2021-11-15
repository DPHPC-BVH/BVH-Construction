#include "CudaBVHBuilder.h"
#include <cuda.h>
#include <cuda_runtime.h>
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

	// Copying bounding boxes to device
	cudaMalloc(&primitiveInfo_device, sizeof(BVHPrimitiveInfoWithIndex) * primitiveInfo.size());
	cudaMemcpy(primitiveInfo_device, primitiveInfo.data(), sizeof(BVHPrimitiveInfoWithIndex) * primitiveInfo.size(), cudaMemcpyHostToDevice);
	// For n primitives (leaf nodes) there are always n-1 interior nodes
	cudaMalloc(&interiorNodes_device, sizeof(BVHBuildNodeDevice) * (primitiveInfo.size()-1));
}


CudaBVHBuilder::~CudaBVHBuilder() {
	cudaFree(primitiveInfo_device);
	cudaFree(interiorNodes_device);
}


void CudaBVHBuilder::BuildBVH() {
	// Launch the kernel to parallelly construct all the interior nodes
	ParallelConstructInteriorNodes << < /* Identify number of threads here*/ >> > (primitiveInfo.size(), primitiveInfo_device, interiorNodes_device);

	// Copy back the interior nodes to CPU
	// Linearize the BVH (see BVHBuilder::FlattenBVHTree) and feed the required data to BVHBuilder::bvh  

}


NAMESPACE_DPHPC_END


