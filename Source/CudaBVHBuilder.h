#pragma once

#include "BVHBuilder.h"


NAMESPACE_DPHPC_BEGIN



class CudaBVHBuilder : public BVHBuilder {
public:
	CudaBVHBuilder(BVH& bvh);

	// Inherited via BVHBuilder
	virtual void BuildBVH() override;

private:
};


NAMESPACE_DPHPC_END
