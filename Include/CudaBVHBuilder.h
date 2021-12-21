#pragma once

#include "BVHBuilder.h"
#include <cuda.h>
#include <cuda_runtime.h>

// Nasty hack such that we can benchmark private functions
#define private public

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
	CudaBVHBuilder(BVH& bvh, bool sharedMemoryUsed = false);
	~CudaBVHBuilder();

	// Inherited via BVHBuilder
	virtual void BuildBVH() override;

private:
	// If true, shared memory is used to compute bounding boxes
	bool sharedMemoryUsed;

	std::vector<BVHPrimitiveInfoWithIndex> primitiveInfo;
	int nPrimitives = 0;

	int FlattenBVHTree(CudaBVHBuildNode nodes[], int nodeIndex, int* offset, int totalPrimitives);




	void GenerateMortonCodesHelper();

	void SortMortonCodesHelper();
	
	void  BuildTreeHierarchyHelper();

	void ComputeBoundingBoxesHelper();

	void PermutePrimitivesAndFlattenTree();

	// Buffers used during construction. Freed after construction.
	BVHPrimitiveInfoWithIndex* dPrimitiveInfo = nullptr;
	unsigned int* dMortonCodes = nullptr;
	unsigned int* dMortonIndices = nullptr;
	unsigned int* dMortonCodesSorted = nullptr;
	unsigned int* dMortonIndicesSorted = nullptr;
	CudaBVHBuildNode* dTree = nullptr;



	void AllocAuxBuffers() {
		// For BVH construction we only need the bounding boxes and the centroids of the primitive
		cudaMalloc(&dPrimitiveInfo, sizeof(BVHPrimitiveInfoWithIndex) * nPrimitives);
		cudaMemcpy(dPrimitiveInfo, primitiveInfo.data(), sizeof(BVHPrimitiveInfoWithIndex) * nPrimitives, cudaMemcpyHostToDevice);
		
		cudaMalloc(&dMortonCodes, sizeof(unsigned int) * nPrimitives);
		cudaMalloc(&dMortonIndices, sizeof(unsigned int) * nPrimitives);

		cudaMalloc(&dMortonCodesSorted, sizeof(unsigned int) * nPrimitives);
		cudaMalloc(&dMortonIndicesSorted, sizeof(unsigned int) * nPrimitives);

		cudaMalloc(&dTree, sizeof(CudaBVHBuildNode) * (2 * nPrimitives - 1));
		
	}

	static void inline safeFree(void* devPtr) {
		if(devPtr) {
			cudaFree(devPtr);
			devPtr = nullptr;
		} 
	}
	void FreeAuxBuffers() {
		safeFree(dPrimitiveInfo);
		safeFree(dMortonCodes);
		safeFree(dMortonIndices);
		safeFree(dMortonCodesSorted);
		safeFree(dMortonIndicesSorted);
		safeFree(dTree);
	}


};


NAMESPACE_DPHPC_END

// Nasty hack such that we can benchmark private functions
#undef private
