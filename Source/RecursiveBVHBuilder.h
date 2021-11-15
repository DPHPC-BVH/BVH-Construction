#pragma once


#include "BVHBuilder.h"

NAMESPACE_DPHPC_BEGIN

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

	std::vector<BVHPrimitiveInfo> primitiveInfo;
	const SplitMethod splitMethod;

};

class LBVHBuilder : public BVHBuilder {
public:

	// Inherited via BVHBuilder
	virtual void BuildBVH() override;
};

NAMESPACE_DPHPC_END
