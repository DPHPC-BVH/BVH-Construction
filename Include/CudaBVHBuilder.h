#pragma once

#include "BVHBuilder.h"
#include <cuda.h>
#include <cuda_runtime.h>


NAMESPACE_DPHPC_BEGIN

// Similar to BVHPrimitiveInfo, with an additional index that points to the element in array BVH::primitives
// We store index because it might be sorted during the construction of the BVH. After BVH is constructed, BVH::primitives will be shuffled accordingly.
struct BVHPrimitiveInfoWithIndex {
	BVHPrimitiveInfoWithIndex() {}
	BVHPrimitiveInfoWithIndex(int idx, const Bounds3f& bounds)
		: idx(idx),
		bounds(bounds),
		centroid(.5f * bounds.pMin + .5f * bounds.pMax) {
	}
	int idx;
	Bounds3f bounds;
	Point3f centroid;
};

// A GPU version of BVHBuildNode.
// We are using indices instead of pointers to allow direct copying of the nodes from device to host.
struct CudaBVHBuildNode {
	CudaBVHBuildNode() : children{-1, -1}, parent(-1), dataIdx(-1) {}
	CudaBVHBuildNode(int left, int right, int parent) : children{left, right}, parent(parent), dataIdx(-1) {}
	CudaBVHBuildNode(int left, int right, int parent, int dataIdx) : children{left, right}, parent(parent), dataIdx(dataIdx) {}

	Bounds3f bounds;
	// The indices specifying the location of the children in an array of CudaBVHBuildNode's.
	// -1 represents no child
	int children[2];

	// The index specifying the location of the parent in an array of CudaBVHBuildNode's.
	// -1 repreesnets no parent, especially the root has parent == -1
	int parent; 

	// For leaf nodes this dataIdx specifies the location of the corresponding primitive in BVH::primitives
	// For interior nodes dataIdx is -1
	int dataIdx;
};

class CudaBVHBuilder : public BVHBuilder {
public:
	CudaBVHBuilder(BVH& bvh);
	~CudaBVHBuilder();

	// Inherited via BVHBuilder
	virtual void BuildBVH() override;

private:
	std::vector<BVHPrimitiveInfoWithIndex> primitiveInfo;
	int FlattenBVHTree(CudaBVHBuildNode nodes[], int nodeIndex, int* offset, int totalPrimitives);

	BVHPrimitiveInfoWithIndex* PrepareDevicePrimitiveInfo(int nPrimitives);

	void GenerateMortonCodesHelper(BVHPrimitiveInfoWithIndex* dPrimitiveInfo, unsigned int** dMortonCodes,
			unsigned int** dMortonIndices, int nPrimitives);

	unsigned int* SortMortonCodesHelper(BVHPrimitiveInfoWithIndex* dPrimitiveInfo, unsigned int* dMortonCodes,
			unsigned int* dMortonIndices, unsigned int** dMortonCodesSorted, unsigned int** dMortonIndicesSorted, int nPrimitives);
	
	CudaBVHBuildNode* BuildTreeHierarchyHelper(unsigned int* dMortonCodesSorted, unsigned int* dMortonIndicesSorted, int nPrimitives);

	CudaBVHBuildNode* ComputeBoundingBoxesHelper(BVHPrimitiveInfoWithIndex* dPrimitiveInfo, CudaBVHBuildNode* dTree, int nPrimitives);

	void PermutePrimitivesAndFlattenTree(unsigned int* indicesSorted, CudaBVHBuildNode* treeWithBoundingBoxes, int nPrimitives);


};


NAMESPACE_DPHPC_END
