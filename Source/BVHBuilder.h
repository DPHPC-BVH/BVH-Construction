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

struct BVHPrimitiveInfo {
	BVHPrimitiveInfo() {}
	BVHPrimitiveInfo(size_t primitiveNumber, const Bounds3f& bounds)
		: primitiveNumber(primitiveNumber),
		bounds(bounds),
		centroid(.5f * bounds.pMin + .5f * bounds.pMax) {
	}
	size_t primitiveNumber;
	Bounds3f bounds;
	Point3f centroid;
};

class BVHBuilder {
public:
	BVHBuilder(BVH& bvh);
	virtual void BuildBVH() = 0;

protected:
	int FlattenBVHTree(BVHBuildNode* node, int* offset);
	BVH& bvh;
	std::vector<BVHPrimitiveInfo> primitiveInfo;
};


class RecursiveBVHBuilder : public BVHBuilder {
public:
	enum class SplitMethod { SAH, HLBVH, Middle, EqualCounts };

	RecursiveBVHBuilder(BVH& bvh, SplitMethod splitMethod = SplitMethod::SAH);

	// Inherited via BVHBuilder
	virtual void BuildBVH() override;

private:
	BVHBuildNode* RecursiveBuild(
		MemoryArena& arena, std::vector<BVHPrimitiveInfo>& primitiveInfo,
		int start, int end, int* totalNodes,
		std::vector<std::shared_ptr<Primitive>>& orderedPrims);

	const SplitMethod splitMethod;
};

class LBVHBuilder : public BVHBuilder {
public:

	// Inherited via BVHBuilder
	virtual void BuildBVH() override;
};

NAMESPACE_DPHPC_END
