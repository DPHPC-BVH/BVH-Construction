#include "BVHBuilder.h"
#include "RecursiveBVHBuilder.h"
#include "CudaBVHBuilder.h"


NAMESPACE_DPHPC_BEGIN


BVHBuilder::BVHBuilder(BVH& bvh) :
	bvh(bvh)
{
}

BVHBuilder* BVHBuilder::MakeBVHBuilder(BVHBuilderType type, BVH* bvh) {
	switch (type) {
		case BVHBuilderType::RecursiveBVHBuilder:
			return new RecursiveBVHBuilder(*bvh);
		case BVHBuilderType::CudaBVHBuilder:
			return new CudaBVHBuilder(*bvh);
		case BVHBuilderType::CudaBVHBuilderWithSharedMemory:
			return new CudaBVHBuilder(*bvh, true);
		default:
			throw "Scene::BuildBVH: Invalid BVHBuilderType";
	}
}

int BVHBuilder::FlattenBVHTree(BVHBuildNode* node, int* offset) {
	LinearBVHNode* linearNode = &bvh.nodes[*offset];
	linearNode->bounds = node->bounds;
	int myOffset = (*offset)++;
	if (node->nPrimitives > 0) {
		CHECK(!node->children[0] && !node->children[1]);
		CHECK_LT(node->nPrimitives, 65536);
		linearNode->primitivesOffset = node->firstPrimOffset;
		linearNode->nPrimitives = node->nPrimitives;
	}
	else {
		// Create interior flattened BVH node
		linearNode->axis = node->splitAxis;
		linearNode->nPrimitives = 0;
		FlattenBVHTree(node->children[0], offset);
		linearNode->secondChildOffset =
			FlattenBVHTree(node->children[1], offset);
	}

	return myOffset;
}

NAMESPACE_DPHPC_END
