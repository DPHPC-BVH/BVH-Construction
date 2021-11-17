#pragma once

#include "DPHPC.h"
#include "Primitive.h"

NAMESPACE_DPHPC_BEGIN
class Triangle : public Primitive {
public:
	Triangle(int faceIndex, const Point3f* vertexBuffer, const int* indexBuffer);


	Bounds3f WorldBound() const override;


	bool Intersect(const Ray& r, Intersection*) const override;


	bool IntersectP(const Ray& r) const override;

private:
	const Point3f* v[3];

};

NAMESPACE_DPHPC_END
