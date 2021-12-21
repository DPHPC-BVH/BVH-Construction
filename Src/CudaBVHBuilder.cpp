#include "CudaBVHBuilder.h"
#include "CudaBVHBuilder.cuh"
#include "CubWrapper.cuh"
#include "Utilities.h"

NAMESPACE_DPHPC_BEGIN


CudaBVHBuilder::CudaBVHBuilder(BVH& bvh) 
	: BVHBuilder(bvh)
{
	// We only need to know each bounding box of the primitive to construct the BVH.
	primitiveInfo.reserve(bvh.primitives.size());
	for (int i = 0; i < bvh.primitives.size(); ++i) {
		primitiveInfo.push_back({ i, bvh.primitives[i]->WorldBound() });
	}
	nPrimitives = primitiveInfo.size();
	int nTasksPerBlock = blockSize * stride;
	gridSize = (nPrimitives + (nTasksPerBlock - 1)) / nTasksPerBlock;
}


CudaBVHBuilder::~CudaBVHBuilder() {
	FreeAuxBuffers();
}


void CudaBVHBuilder::BuildBVH() {

	AllocAuxBuffers();

	// 1. Compute Morton Codes
	unsigned int* dMortonCodes;
	unsigned int* dMortonIndices;
	GenerateMortonCodesHelper();
	
	// 2. Sort Morton Codes
	SortMortonCodesHelper();

	// 3. Build tree hierarchy of CudaBVHBuildNodes
	BuildTreeHierarchyHelper();

	// 4. Compute Bounding Boxes of each node
	ComputeBoundingBoxesHelper();

	// 5. Flatten Tree and order BVH::primitives according to dMortonIndicesSorted
	// Remarks: We could maybe do this more efficient in GPU?
	PermutePrimitivesAndFlattenTree();


}

//void CudaBVHBuilder::SetStride(int stride) {
//	this->stride = stride;
//}
//
//void CudaBVHBuilder::SetOccupancy(float occupancy, int nPrimitives) {
//	// Query device properties
//	cudaDeviceProp prop;
//	cudaGetDeviceProperties(&prop, 0);
//	// For RTX 2060 (Turing arthitecture), multiProcessorCount=30, maxThreadsPerMultiProcessor=1024, warpSize=32
//	int maxWarps = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / prop.warpSize);
//	int nWarps = std::roundf(maxWarps * occupancy);
//	int nThreads = nWarps * prop.warpSize;
//	this->stride = (nPrimitives + (nThreads - 1)) / nThreads;
//	//std::cout << "occupancy: " << occupancy * 100 << "stride: " << stride << std::endl;
//}

void CudaBVHBuilder::SetBlockNum(int blockNum) {
	if (blockNum <= 0) {
		return;
	}
	int nThreads = blockSize * blockNum;
	stride = (nPrimitives + (nThreads - 1)) / nThreads;
	gridSize = blockNum;
}


void CudaBVHBuilder::GenerateMortonCodesHelper() {

	GenerateMortonCodes32(nPrimitives, gridSize, stride, dPrimitiveInfo, dMortonCodes, dMortonIndices);
}

void CudaBVHBuilder::SortMortonCodesHelper() {
	DeviceSort(nPrimitives, &dMortonCodes, &dMortonCodesSorted,
				&dMortonIndices, &dMortonIndicesSorted);

}

void CudaBVHBuilder::BuildTreeHierarchyHelper() {
	BuildTreeHierarchy(nPrimitives, gridSize, stride, dMortonCodesSorted, dMortonIndicesSorted, dTree);
}

void CudaBVHBuilder::ComputeBoundingBoxesHelper() {
	
	ComputeBoundingBoxes(nPrimitives, gridSize, stride, dTree, dPrimitiveInfo);
}

void CudaBVHBuilder::PermutePrimitivesAndFlattenTree() {
	
	unsigned int* indicesSorted = (unsigned int*) malloc(nPrimitives * sizeof(unsigned int));
	cudaMemcpy(indicesSorted, dMortonIndicesSorted, nPrimitives * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	CudaBVHBuildNode* treeWithBoundingBoxes = (CudaBVHBuildNode*) malloc((2 * nPrimitives - 1) * sizeof(CudaBVHBuildNode));
	cudaMemcpy(treeWithBoundingBoxes, dTree, (2 * nPrimitives - 1) * sizeof(CudaBVHBuildNode), cudaMemcpyDeviceToHost);
	
	applyPermutation(bvh.primitives, indicesSorted, nPrimitives);
	bvh.nodes = AllocAligned<LinearBVHNode>(2 * nPrimitives - 1);
	int offset = 0;
	FlattenBVHTree(treeWithBoundingBoxes, 0, &offset, nPrimitives);

	free(treeWithBoundingBoxes);
	free(indicesSorted);
}


int CudaBVHBuilder::FlattenBVHTree(CudaBVHBuildNode* nodes, int nodeIndex, int* offset, int totalPrimitives) {
	LinearBVHNode* linearNode = &bvh.nodes[*offset];
	CudaBVHBuildNode node = nodes[nodeIndex];
	linearNode->bounds = node.bounds;
	int myOffset = (*offset)++;
	if (node.children[0] == -1 && node.children[1] == -1) {
		linearNode->primitivesOffset = nodeIndex - (totalPrimitives - 1);
		linearNode->nPrimitives = 1;
	}
	else {
		// Create interior flattened BVH node
		linearNode->axis = 0;
		linearNode->nPrimitives = 0;
		FlattenBVHTree(nodes, node.children[0], offset, totalPrimitives);
		linearNode->secondChildOffset =
			FlattenBVHTree(nodes, node.children[1], offset, totalPrimitives);
	}

	return myOffset;
}


NAMESPACE_DPHPC_END


