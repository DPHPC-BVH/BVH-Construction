#include "RecursiveBVHBuilder.h"
#include <algorithm>

NAMESPACE_DPHPC_BEGIN

RecursiveBVHBuilder::RecursiveBVHBuilder(BVH& bvh, SplitMethod splitMethod /*= SplitMethod::SAH*/)
	: BVHBuilder(bvh),
	primitiveInfo(bvh.primitives.size()),
	splitMethod(splitMethod)
{
	for (size_t i = 0; i < primitiveInfo.size(); ++i)
		primitiveInfo[i] = { i, bvh.primitives[i]->WorldBound() };
}

void RecursiveBVHBuilder::BuildBVH() {
	MemoryArena arena(1024 * 1024);
	int totalNodes = 0;
	std::vector<std::shared_ptr<Primitive>> orderedPrims;
	orderedPrims.reserve(bvh.primitives.size());
	BVHBuildNode* root = RecursiveBuild(arena, primitiveInfo, 0, bvh.primitives.size(),
		&totalNodes, orderedPrims);
	bvh.primitives.swap(orderedPrims);

	bvh.nodes = AllocAligned<LinearBVHNode>(totalNodes);
	int offset = 0;
	FlattenBVHTree(root, &offset);
	CHECK_EQ(totalNodes, offset);
}

struct BucketInfo {
	int count = 0;
	Bounds3f bounds;
};

BVHBuildNode* RecursiveBVHBuilder::RecursiveBuild(
	MemoryArena& arena, std::vector<BVHPrimitiveInfo>& primitiveInfo, int start,
	int end, int* totalNodes,
	std::vector<std::shared_ptr<Primitive>>& orderedPrims) {
	CHECK_NE(start, end);
	BVHBuildNode* node = arena.Alloc<BVHBuildNode>();
	(*totalNodes)++;
	// Compute bounds of all primitives in BVH node
	Bounds3f bounds;
	for (int i = start; i < end; ++i)
		bounds = Union(bounds, primitiveInfo[i].bounds);
	int nPrimitives = end - start;
	if (nPrimitives == 1) {
		// Create leaf _BVHBuildNode_
		int firstPrimOffset = orderedPrims.size();
		for (int i = start; i < end; ++i) {
			int primNum = primitiveInfo[i].primitiveNumber;
			orderedPrims.push_back(bvh.primitives[primNum]);
		}
		node->InitLeaf(firstPrimOffset, nPrimitives, bounds);
		return node;
	}
	else {
		// Compute bound of primitive centroids, choose split dimension _dim_
		Bounds3f centroidBounds;
		for (int i = start; i < end; ++i)
			centroidBounds = Union(centroidBounds, primitiveInfo[i].centroid);
		int dim = centroidBounds.MaximumExtent();

		// Partition primitives into two sets and build children
		int mid = (start + end) / 2;
		if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
			// Create leaf _BVHBuildNode_
			int firstPrimOffset = orderedPrims.size();
			for (int i = start; i < end; ++i) {
				int primNum = primitiveInfo[i].primitiveNumber;
				orderedPrims.push_back(bvh.primitives[primNum]);
			}
			node->InitLeaf(firstPrimOffset, nPrimitives, bounds);
			return node;
		}
		else {
			// Partition primitives based on _splitMethod_
			switch (splitMethod) {
			case SplitMethod::Middle: {
				// Partition primitives through node's midpoint
				Float pmid =
					(centroidBounds.pMin[dim] + centroidBounds.pMax[dim]) / 2;
				BVHPrimitiveInfo* midPtr = std::partition(
					&primitiveInfo[start], &primitiveInfo[end - 1] + 1,
					[dim, pmid](const BVHPrimitiveInfo& pi) {
					return pi.centroid[dim] < pmid;
				});
				mid = midPtr - &primitiveInfo[0];
				// For lots of prims with large overlapping bounding boxes, this
				// may fail to partition; in that case don't break and fall
				// through
				// to EqualCounts.
				if (mid != start && mid != end) break;
			}
			case SplitMethod::EqualCounts: {
				// Partition primitives into equally-sized subsets
				mid = (start + end) / 2;
				std::nth_element(&primitiveInfo[start], &primitiveInfo[mid],
					&primitiveInfo[end - 1] + 1,
					[dim](const BVHPrimitiveInfo& a,
						const BVHPrimitiveInfo& b) {
					return a.centroid[dim] < b.centroid[dim];
				});
				break;
			}
			case SplitMethod::SAH:
			default: {
				// Partition primitives using approximate SAH
				if (nPrimitives <= 2) {
					// Partition primitives into equally-sized subsets
					mid = (start + end) / 2;
					std::nth_element(&primitiveInfo[start], &primitiveInfo[mid],
						&primitiveInfo[end - 1] + 1,
						[dim](const BVHPrimitiveInfo& a,
							const BVHPrimitiveInfo& b) {
						return a.centroid[dim] <
							b.centroid[dim];
					});
				}
				else {
					// Allocate _BucketInfo_ for SAH partition buckets
					constexpr int nBuckets = 12;
					BucketInfo buckets[nBuckets];

					// Initialize _BucketInfo_ for SAH partition buckets
					for (int i = start; i < end; ++i) {
						int b = nBuckets *
							centroidBounds.Offset(
								primitiveInfo[i].centroid)[dim];
						if (b == nBuckets) b = nBuckets - 1;
						CHECK_GE(b, 0);
						CHECK_LT(b, nBuckets);
						buckets[b].count++;
						buckets[b].bounds =
							Union(buckets[b].bounds, primitiveInfo[i].bounds);
					}

					// Compute costs for splitting after each bucket
					Float cost[nBuckets - 1];
					for (int i = 0; i < nBuckets - 1; ++i) {
						Bounds3f b0, b1;
						int count0 = 0, count1 = 0;
						for (int j = 0; j <= i; ++j) {
							b0 = Union(b0, buckets[j].bounds);
							count0 += buckets[j].count;
						}
						for (int j = i + 1; j < nBuckets; ++j) {
							b1 = Union(b1, buckets[j].bounds);
							count1 += buckets[j].count;
						}
						cost[i] = 1 +
							(count0 * b0.SurfaceArea() +
								count1 * b1.SurfaceArea()) /
							bounds.SurfaceArea();
					}

					// Find bucket to split at that minimizes SAH metric
					Float minCost = cost[0];
					int minCostSplitBucket = 0;
					for (int i = 1; i < nBuckets - 1; ++i) {
						if (cost[i] < minCost) {
							minCost = cost[i];
							minCostSplitBucket = i;
						}
					}

					// Either create leaf or split primitives at selected SAH
					// bucket
					Float leafCost = nPrimitives;
					if (nPrimitives > /*maxPrimsInNode*/1 || minCost < leafCost) {
						BVHPrimitiveInfo* pmid = std::partition(
							&primitiveInfo[start], &primitiveInfo[end - 1] + 1,
							[=](const BVHPrimitiveInfo& pi) {
							int b = nBuckets *
								centroidBounds.Offset(pi.centroid)[dim];
							if (b == nBuckets) b = nBuckets - 1;
							CHECK_GE(b, 0);
							CHECK_LT(b, nBuckets);
							return b <= minCostSplitBucket;
						});
						mid = pmid - &primitiveInfo[0];
					}
					else {
						// Create leaf _BVHBuildNode_
						int firstPrimOffset = orderedPrims.size();
						for (int i = start; i < end; ++i) {
							int primNum = primitiveInfo[i].primitiveNumber;
							orderedPrims.push_back(bvh.primitives[primNum]);
						}
						node->InitLeaf(firstPrimOffset, nPrimitives, bounds);
						return node;
					}
				}
				break;
			}
			}
			node->InitInterior(dim,
				RecursiveBuild(arena, primitiveInfo, start, mid,
					totalNodes, orderedPrims),
				RecursiveBuild(arena, primitiveInfo, mid, end,
					totalNodes, orderedPrims));
		}
	}
	return node;
}

NAMESPACE_DPHPC_END
