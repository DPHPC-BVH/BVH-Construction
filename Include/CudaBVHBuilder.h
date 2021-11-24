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

	Bounds3f bounds;
	// The indies specifing the location of the children in an array of CudaBVHBuildNode's.
	// -1 represents no child
	int children[2];

	// The indies specifing the location of the parent in an array of CudaBVHBuildNode's.
	// -1 repreesnets no parent, especially the root has parent == -1
	int parent; 

};

class CudaBVHBuilder : public BVHBuilder {
public:
	CudaBVHBuilder(BVH& bvh);
	~CudaBVHBuilder();

	// Inherited via BVHBuilder
	virtual void BuildBVH() override;

private:
	std::vector<BVHPrimitiveInfoWithIndex> primitiveInfo;
	uint32_t* mortonCodes;

};


NAMESPACE_DPHPC_END
