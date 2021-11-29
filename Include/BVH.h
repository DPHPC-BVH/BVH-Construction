/*
    Copyright notice:
    This project is based on pbrt, with necessary modifications.
    Any files that are originally part of pbrt will contain the following copyright notice.
*/

/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#pragma once


#include "DPHPC.h"
#include "Primitive.h"
#include <atomic>


NAMESPACE_DPHPC_BEGIN




struct LinearBVHNode {
	Bounds3f bounds;
	union {
		int primitivesOffset;   // leaf
		int secondChildOffset;  // interior
	};
	uint16_t nPrimitives;  // 0 -> interior node
	uint8_t axis;          // interior node: xyz
	uint8_t pad[1];        // ensure 32 byte total size
};

class BVH {
    friend class BVHBuilder;
    friend class RecursiveBVHBuilder;
    friend class LBVHBuilder;
    friend class CudaBVHBuilder;
public:
    BVH();
	BVH(std::vector<std::shared_ptr<Primitive>> p);
    ~BVH();

	Bounds3f WorldBound() const;

	bool Intersect(const Ray& ray, Intersection* isect) const;
	bool IntersectP(const Ray& ray) const;

private:
    bool bIsBuilt;

	std::vector<std::shared_ptr<Primitive>> primitives;
	LinearBVHNode* nodes = nullptr;
};


NAMESPACE_DPHPC_END



