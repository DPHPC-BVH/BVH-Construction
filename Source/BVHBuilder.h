#pragma once



#include "DPHPC.h"
#include "BVH.h"
#include "Memory.h"

NAMESPACE_DPHPC_BEGIN

// Forward Declarations
struct BVHBuildNode;
struct BVHPrimitiveInfo;

class BVHBuilder {
public:
	BVHBuilder(BVH& bvh);
	virtual void BuildBVH() = 0;

private:
	int FlattenBVHTree(BVHBuildNode* node, int* offset);
	BVH& bvh;
	std::vector<BVHPrimitiveInfo> primitiveInfo;
};


class RecursiveBVHBuilder : public BVHBuilder {
public:
	enum class SplitMethod { SAH, HLBVH, Middle, EqualCounts };

	RecursiveBVHBuilder(SplitMethod splitMethod = SplitMethod::SAH);

	// Inherited via BVHBuilder
	virtual void BuildBVH(BVH& bvh) override;

private:
	BVHBuildNode* RecursiveBuild(
		MemoryArena& arena, std::vector<BVHPrimitiveInfo>& primitiveInfo,
		int start, int end, int* totalNodes,
		std::vector<std::shared_ptr<Primitive>>& orderedPrims);

	const SplitMethod splitMethod;
};

class LBVHBuilder : public BVHBuilder {
public:
	LBVHBuilder();

	// Inherited via BVHBuilder
	virtual void BuildBVH(BVH& bvh) override;
};

NAMESPACE_DPHPC_END
