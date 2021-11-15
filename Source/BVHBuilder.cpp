#include "BVHBuilder.h"

NAMESPACE_DPHPC_BEGIN


BVHBuilder::BVHBuilder(BVH& bvh) :
	bvh(bvh),
	primitiveInfo(bvh.primitives.size()) 
{
	for (size_t i = 0; i < primitiveInfo.size(); ++i)
		primitiveInfo[i] = { i, bvh.primitives[i]->WorldBound() };

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
