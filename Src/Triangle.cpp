#include "Triangle.h"

NAMESPACE_DPHPC_BEGIN



Triangle::Triangle(int faceIndex, const Point3f* vertexBuffer, const int* indexBuffer) {
	v[0] = vertexBuffer + indexBuffer[3 * faceIndex];
	v[1] = vertexBuffer + indexBuffer[3 * faceIndex + 1];
	v[2] = vertexBuffer + indexBuffer[3 * faceIndex + 2];
}

Bounds3f Triangle::WorldBound() const {
	return Union(Bounds3f(*v[0], *v[1]),*v[2]);
}

bool Triangle::Intersect(const Ray& ray, Intersection* isect) const {
	// Get triangle vertices in _p0_, _p1_, and _p2_
	const Point3f& p0 = *(v[0]);
	const Point3f& p1 = *(v[1]);
	const Point3f& p2 = *(v[2]);

	// Perform ray--triangle intersection test

	// Transform triangle vertices to ray coordinate space

	// Translate vertices based on ray origin
	Point3f p0t = p0 - Vector3f(ray.o);
	Point3f p1t = p1 - Vector3f(ray.o);
	Point3f p2t = p2 - Vector3f(ray.o);

	// Permute components of triangle vertices and ray direction
	int kz = MaxDimension(Abs(ray.d));
	int kx = kz + 1;
	if (kx == 3) kx = 0;
	int ky = kx + 1;
	if (ky == 3) ky = 0;
	Vector3f d = Permute(ray.d, kx, ky, kz);
	p0t = Permute(p0t, kx, ky, kz);
	p1t = Permute(p1t, kx, ky, kz);
	p2t = Permute(p2t, kx, ky, kz);

	// Apply shear transformation to translated vertex positions
	Float Sx = -d.x / d.z;
	Float Sy = -d.y / d.z;
	Float Sz = 1.f / d.z;
	p0t.x += Sx * p0t.z;
	p0t.y += Sy * p0t.z;
	p1t.x += Sx * p1t.z;
	p1t.y += Sy * p1t.z;
	p2t.x += Sx * p2t.z;
	p2t.y += Sy * p2t.z;

	// Compute edge function coefficients _e0_, _e1_, and _e2_
	Float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
	Float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
	Float e2 = p0t.x * p1t.y - p0t.y * p1t.x;

	// Fall back to double precision test at triangle edges
	if (sizeof(Float) == sizeof(float) &&
		(e0 == 0.0f || e1 == 0.0f || e2 == 0.0f)) {
		double p2txp1ty = (double)p2t.x * (double)p1t.y;
		double p2typ1tx = (double)p2t.y * (double)p1t.x;
		e0 = (float)(p2typ1tx - p2txp1ty);
		double p0txp2ty = (double)p0t.x * (double)p2t.y;
		double p0typ2tx = (double)p0t.y * (double)p2t.x;
		e1 = (float)(p0typ2tx - p0txp2ty);
		double p1txp0ty = (double)p1t.x * (double)p0t.y;
		double p1typ0tx = (double)p1t.y * (double)p0t.x;
		e2 = (float)(p1typ0tx - p1txp0ty);
	}

	// Perform triangle edge and determinant tests
	if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
		return false;
	Float det = e0 + e1 + e2;
	if (det == 0) return false;

	// Compute scaled hit distance to triangle and test against ray $t$ range
	p0t.z *= Sz;
	p1t.z *= Sz;
	p2t.z *= Sz;
	Float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
	if (det < 0 && (tScaled >= 0 || tScaled < ray.tMax * det))
		return false;
	else if (det > 0 && (tScaled <= 0 || tScaled > ray.tMax * det))
		return false;

	// Compute barycentric coordinates and $t$ value for triangle intersection
	Float invDet = 1 / det;
	Float b0 = e0 * invDet;
	Float b1 = e1 * invDet;
	Float b2 = e2 * invDet;
	Float t = tScaled * invDet;

	// Ensure that computed triangle $t$ is conservatively greater than zero

	// Compute $\delta_z$ term for triangle $t$ error bounds
	Float maxZt = MaxComponent(Abs(Vector3f(p0t.z, p1t.z, p2t.z)));
	Float deltaZ = gamma(3) * maxZt;

	// Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
	Float maxXt = MaxComponent(Abs(Vector3f(p0t.x, p1t.x, p2t.x)));
	Float maxYt = MaxComponent(Abs(Vector3f(p0t.y, p1t.y, p2t.y)));
	Float deltaX = gamma(5) * (maxXt + maxZt);
	Float deltaY = gamma(5) * (maxYt + maxZt);

	// Compute $\delta_e$ term for triangle $t$ error bounds
	Float deltaE =
		2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

	// Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
	Float maxE = MaxComponent(Abs(Vector3f(e0, e1, e2)));
	Float deltaT = 3 *
		(gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) *
		std::abs(invDet);
	if (t <= deltaT) return false;



	// Interpolate $(u,v)$ parametric coordinates and hit point
	Point3f pHit = b0 * p0 + b1 * p1 + b2 * p2;


	// Fill in _SurfaceInteraction_ from triangle hit
	*isect = Intersection{ pHit, Normal3f(Normalize(Cross(p0 - p2, p1 - p2))) };

	return true;
}

bool Triangle::IntersectP(const Ray& ray) const {
	// Get triangle vertices in _p0_, _p1_, and _p2_
	const Point3f& p0 = *v[0];
	const Point3f& p1 = *v[1];
	const Point3f& p2 = *v[2];

	// Perform ray--triangle intersection test

	// Transform triangle vertices to ray coordinate space

	// Translate vertices based on ray origin
	Point3f p0t = p0 - Vector3f(ray.o);
	Point3f p1t = p1 - Vector3f(ray.o);
	Point3f p2t = p2 - Vector3f(ray.o);

	// Permute components of triangle vertices and ray direction
	int kz = MaxDimension(Abs(ray.d));
	int kx = kz + 1;
	if (kx == 3) kx = 0;
	int ky = kx + 1;
	if (ky == 3) ky = 0;
	Vector3f d = Permute(ray.d, kx, ky, kz);
	p0t = Permute(p0t, kx, ky, kz);
	p1t = Permute(p1t, kx, ky, kz);
	p2t = Permute(p2t, kx, ky, kz);

	// Apply shear transformation to translated vertex positions
	Float Sx = -d.x / d.z;
	Float Sy = -d.y / d.z;
	Float Sz = 1.f / d.z;
	p0t.x += Sx * p0t.z;
	p0t.y += Sy * p0t.z;
	p1t.x += Sx * p1t.z;
	p1t.y += Sy * p1t.z;
	p2t.x += Sx * p2t.z;
	p2t.y += Sy * p2t.z;

	// Compute edge function coefficients _e0_, _e1_, and _e2_
	Float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
	Float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
	Float e2 = p0t.x * p1t.y - p0t.y * p1t.x;

	// Fall back to double precision test at triangle edges
	if (sizeof(Float) == sizeof(float) &&
		(e0 == 0.0f || e1 == 0.0f || e2 == 0.0f)) {
		double p2txp1ty = (double)p2t.x * (double)p1t.y;
		double p2typ1tx = (double)p2t.y * (double)p1t.x;
		e0 = (float)(p2typ1tx - p2txp1ty);
		double p0txp2ty = (double)p0t.x * (double)p2t.y;
		double p0typ2tx = (double)p0t.y * (double)p2t.x;
		e1 = (float)(p0typ2tx - p0txp2ty);
		double p1txp0ty = (double)p1t.x * (double)p0t.y;
		double p1typ0tx = (double)p1t.y * (double)p0t.x;
		e2 = (float)(p1typ0tx - p1txp0ty);
	}

	// Perform triangle edge and determinant tests
	if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
		return false;
	Float det = e0 + e1 + e2;
	if (det == 0) return false;

	// Compute scaled hit distance to triangle and test against ray $t$ range
	p0t.z *= Sz;
	p1t.z *= Sz;
	p2t.z *= Sz;
	Float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
	if (det < 0 && (tScaled >= 0 || tScaled < ray.tMax * det))
		return false;
	else if (det > 0 && (tScaled <= 0 || tScaled > ray.tMax * det))
		return false;

	// Compute barycentric coordinates and $t$ value for triangle intersection
	Float invDet = 1 / det;
	Float b0 = e0 * invDet;
	Float b1 = e1 * invDet;
	Float b2 = e2 * invDet;
	Float t = tScaled * invDet;

	// Ensure that computed triangle $t$ is conservatively greater than zero

	// Compute $\delta_z$ term for triangle $t$ error bounds
	Float maxZt = MaxComponent(Abs(Vector3f(p0t.z, p1t.z, p2t.z)));
	Float deltaZ = gamma(3) * maxZt;

	// Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
	Float maxXt = MaxComponent(Abs(Vector3f(p0t.x, p1t.x, p2t.x)));
	Float maxYt = MaxComponent(Abs(Vector3f(p0t.y, p1t.y, p2t.y)));
	Float deltaX = gamma(5) * (maxXt + maxZt);
	Float deltaY = gamma(5) * (maxYt + maxZt);

	// Compute $\delta_e$ term for triangle $t$ error bounds
	Float deltaE =
		2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

	// Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
	Float maxE = MaxComponent(Abs(Vector3f(e0, e1, e2)));
	Float deltaT = 3 *
		(gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) *
		std::abs(invDet);
	if (t <= deltaT) return false;

	return true;
}

NAMESPACE_DPHPC_END