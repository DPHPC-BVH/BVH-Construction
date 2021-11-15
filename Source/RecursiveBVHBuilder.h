#pragma once


#include "BVHBuilder.h"

NAMESPACE_DPHPC_BEGIN


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
