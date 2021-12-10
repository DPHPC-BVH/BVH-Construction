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
}


CudaBVHBuilder::~CudaBVHBuilder() {
}


void CudaBVHBuilder::BuildBVH() {

	// For BVH construction we only need the bounding boxes and the centroids of the primitive
	const unsigned int nPrimitives = primitiveInfo.size();
	BVHPrimitiveInfoWithIndex* dPrimitiveInfo = CudaBVHBuilder::PrepareDevicePrimitiveInfo(nPrimitives);

	// 1. Compute Morton Codes
	unsigned int* dMortonCodes;
	unsigned int* dMortonIndices;
	CudaBVHBuilder::GenerateMortonCodesHelper(dPrimitiveInfo, &dMortonCodes, &dMortonIndices, nPrimitives);
	
	// 2. Sort Morton Codes
	unsigned int* dMortonCodesSorted;
	unsigned int* dMortonIndicesSorted;
	unsigned int* indicesSorted = CudaBVHBuilder::SortMortonCodesHelper(dPrimitiveInfo, dMortonCodes, dMortonIndices, &dMortonCodesSorted, &dMortonIndicesSorted, nPrimitives);

	// 3. Build tree hierarchy of CudaBVHBuildNodes
	CudaBVHBuildNode* dTree = CudaBVHBuilder::BuildTreeHierarchyHelper(dMortonCodesSorted, dMortonIndicesSorted, nPrimitives);

	// 4. Compute Bounding Boxes of each node
	CudaBVHBuildNode* treeWithBoundingBoxes = CudaBVHBuilder::ComputeBoundingBoxesHelper(dPrimitiveInfo, dTree, nPrimitives);

	// 5. Flatten Tree and order BVH::primitives according to dMortonIndicesSorted
	// Remarks: We could maybe do this more efficient in GPU?
	CudaBVHBuilder::PermutePrimitivesAndFlattenTree(indicesSorted, treeWithBoundingBoxes, nPrimitives);


}

BVHPrimitiveInfoWithIndex* CudaBVHBuilder::PrepareDevicePrimitiveInfo(int nPrimitives) {
	BVHPrimitiveInfoWithIndex* dPrimitiveInfo;
	cudaMalloc(&dPrimitiveInfo, sizeof(BVHPrimitiveInfoWithIndex) * nPrimitives);
	cudaMemcpy(dPrimitiveInfo, primitiveInfo.data(), sizeof(BVHPrimitiveInfoWithIndex) * nPrimitives, cudaMemcpyHostToDevice);
	return dPrimitiveInfo;
}

void CudaBVHBuilder::GenerateMortonCodesHelper(BVHPrimitiveInfoWithIndex* dPrimitiveInfo, unsigned int** dMortonCodes,
		unsigned int** dMortonIndices, int nPrimitives) {
	cudaMalloc(dMortonCodes, sizeof(unsigned int) * nPrimitives);
	cudaMalloc(dMortonIndices, sizeof(unsigned int) * nPrimitives);
	GenerateMortonCodes32(nPrimitives, dPrimitiveInfo, *dMortonCodes, *dMortonIndices);
}

unsigned int* CudaBVHBuilder::SortMortonCodesHelper(BVHPrimitiveInfoWithIndex* dPrimitiveInfo, unsigned int* dMortonCodes,
		unsigned int* dMortonIndices, unsigned int** dMortonCodesSorted, unsigned int** dMortonIndicesSorted, int nPrimitives) {
	
	cudaMalloc(dMortonCodesSorted, sizeof(unsigned int) * nPrimitives);
	cudaMalloc(dMortonIndicesSorted, sizeof(unsigned int) * nPrimitives);
	DeviceSort(nPrimitives, &dMortonCodes, dMortonCodesSorted,
				&dMortonIndices, dMortonIndicesSorted);
	unsigned int* indicesSorted = (unsigned int*) malloc(nPrimitives * sizeof(unsigned int));
	cudaMemcpy(indicesSorted, *dMortonIndicesSorted, nPrimitives * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaFree(dMortonCodes);
	cudaFree(dMortonIndices);

	return indicesSorted;

}

CudaBVHBuildNode* CudaBVHBuilder::BuildTreeHierarchyHelper(unsigned int* dMortonCodesSorted,
		unsigned int* dMortonIndicesSorted, int nPrimitives) {
	
	CudaBVHBuildNode* dTree;
	cudaMalloc(&dTree, sizeof(CudaBVHBuildNode) * (2 * nPrimitives - 1));
	BuildTreeHierarchy(nPrimitives, dMortonCodesSorted, dMortonIndicesSorted, dTree);
	cudaFree(dMortonCodesSorted);
	cudaFree(dMortonIndicesSorted);
	return dTree;
}

CudaBVHBuildNode* CudaBVHBuilder::ComputeBoundingBoxesHelper(BVHPrimitiveInfoWithIndex* dPrimitiveInfo, CudaBVHBuildNode* dTree, int nPrimitives) {
	
	ComputeBoundingBoxes(nPrimitives, dTree, dPrimitiveInfo);
	CudaBVHBuildNode* treeWithBoundingBoxes = (CudaBVHBuildNode*) malloc((2 * nPrimitives - 1) * sizeof(CudaBVHBuildNode));
	cudaMemcpy(treeWithBoundingBoxes, dTree, (2 * nPrimitives - 1) * sizeof(CudaBVHBuildNode), cudaMemcpyDeviceToHost);
	cudaFree(dPrimitiveInfo);
	cudaFree(dTree);

	return treeWithBoundingBoxes;
}

void CudaBVHBuilder::PermutePrimitivesAndFlattenTree(unsigned int* indicesSorted, CudaBVHBuildNode* treeWithBoundingBoxes, int nPrimitives) {
	
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


