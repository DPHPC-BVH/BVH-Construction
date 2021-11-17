#pragma once

#include "BVHBuilder.h"


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
struct BVHBuildNodeDevice {
	int idx;  // The index of this node in the array of all nodes.
	Bounds3f bounds;
	int children[2];  // The indices of its two children. 
	// And more info...
};

class CudaBVHBuilder : public BVHBuilder {
public:
	CudaBVHBuilder(BVH& bvh);
	~CudaBVHBuilder();

	// Inherited via BVHBuilder
	virtual void BuildBVH() override;

private:
	std::vector<BVHPrimitiveInfoWithIndex> primitiveInfo;
	BVHPrimitiveInfoWithIndex* primitiveInfo_device;
	BVHBuildNodeDevice* interiorNodes_device;


};


NAMESPACE_DPHPC_END
