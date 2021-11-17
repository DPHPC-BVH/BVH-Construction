#pragma once
#include "DPHPC.h"

NAMESPACE_DPHPC_BEGIN
// Low Discrepancy Inline Functions
inline uint32_t ReverseBits32(uint32_t n) {
	n = (n << 16) | (n >> 16);
	n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
	n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
	n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
	n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
	return n;
}

inline uint64_t ReverseBits64(uint64_t n) {
	uint64_t n0 = ReverseBits32((uint32_t)n);
	uint64_t n1 = ReverseBits32((uint32_t)(n >> 32));
	return (n0 << 32) | n1;
}

static const float FloatOneMinusEpsilon = 0x1.fffffep-1;
static const Float OneMinusEpsilon = FloatOneMinusEpsilon;

// Low Discrepancy Static Functions
template <int base>
static Float RadicalInverseSpecialized(uint64_t a) {
	const Float invBase = (Float)1 / (Float)base;
	uint64_t reversedDigits = 0;
	Float invBaseN = 1;
	while (a) {
		uint64_t next = a / base;
		uint64_t digit = a - next * base;
		reversedDigits = reversedDigits * base + digit;
		invBaseN *= invBase;
		a = next;
	}
	DCHECK_LT(reversedDigits * invBaseN, 1.00001);
	return std::min(reversedDigits * invBaseN, OneMinusEpsilon);
}

Float RadicalInverse(int baseIndex, uint64_t a) {
	switch (baseIndex) {
	case 0:
		// Compute base-2 radical inverse
		return ReverseBits64(a) * 0x1p-64;
	case 1:
		return RadicalInverseSpecialized<3>(a);
	case 2:
		return RadicalInverseSpecialized<5>(a);
	case 3:
		return RadicalInverseSpecialized<7>(a);
		// Remainder of cases for _RadicalInverse()_
	default:
		CHECK(0); // We only need 2-D samples at most...
		return 0;
	}
}

Point2f ConcentricSampleDisk(const Point2f& u) {
	// Map uniform random numbers to $[-1,1]^2$
	Point2f uOffset = 2.f * u - Vector2f(1, 1);

	// Handle degeneracy at the origin
	if (uOffset.x == 0 && uOffset.y == 0) return Point2f(0, 0);

	// Apply concentric mapping to point
	Float theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
		r = uOffset.x;
		theta = PiOver4 * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
	}
	return r * Point2f(std::cos(theta), std::sin(theta));
}

Vector3f UniformSampleHemisphere(const Point2f& u) {
	Float z = u[0];
	Float r = std::sqrt(std::max((Float)0, (Float)1. - z * z));
	Float phi = 2 * Pi * u[1];
	return Vector3f(r * std::cos(phi), r * std::sin(phi), z);
}

Float UniformHemispherePdf() { return Inv2Pi; }

inline Vector3f CosineSampleHemisphere(const Point2f& u) {
	Point2f d = ConcentricSampleDisk(u);
	Float z = std::sqrt(std::max((Float)0, 1 - d.x * d.x - d.y * d.y));
	return Vector3f(d.x, d.y, z);
}

inline Float CosineHemispherePdf(Float cosTheta) { return cosTheta * InvPi; }


NAMESPACE_DPHPC_END