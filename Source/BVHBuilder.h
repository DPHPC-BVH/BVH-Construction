#pragma once



#include "DPHPC.h"
#include "BVH.h"
#include "Memory.h"

NAMESPACE_DPHPC_BEGIN

struct BVHBuildNode {
	// BVHBuildNode Public Methods
	void InitLeaf(int first, int n, const Bounds3f& b) {
		firstPrimOffset = first;
		nPrimitives = n;
		bounds = b;
		children[0] = children[1] = nullptr;
		/* todo: statistics
		++leafNodes;
		++totalLeafNodes;
		totalPrimitives += n;
		*/
	}
	void InitInterior(int axis, BVHBuildNode* c0, BVHBuildNode* c1) {
		children[0] = c0;
		children[1] = c1;
		bounds = Union(c0->bounds, c1->bounds);
		splitAxis = axis;
		nPrimitives = 0;
		/* todo: statistics
		++interiorNodes;
		*/
	}
	Bounds3f bounds;
	BVHBuildNode* children[2];
	int splitAxis, firstPrimOffset, nPrimitives;
};



class BVHBuilder {
public:
	BVHBuilder(BVH& bvh);
	virtual void BuildBVH() = 0;

protected:
	int FlattenBVHTree(BVHBuildNode* node, int* offset);
	BVH& bvh;

};


NAMESPACE_DPHPC_END
