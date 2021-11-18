#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <limits>
#include <cmath>
#include <list>
#include <cstring>
#include <stdarg.h>

typedef float Float;

// Namespace macros
#define NAMESPACE_BEGIN(name) namespace name{
#define NAMESPACE_END };
#define NAMESPACE_DPHPC_BEGIN NAMESPACE_BEGIN(DPHPC)
#define NAMESPACE_DPHPC_END NAMESPACE_END

// Global Constants
static constexpr Float MaxFloat = std::numeric_limits<Float>::max();
static constexpr Float Infinity = std::numeric_limits<Float>::infinity();
static constexpr Float MachineEpsilon = std::numeric_limits<Float>::epsilon() * 0.5;
static constexpr Float ShadowEpsilon = 0.0001f;
static constexpr Float Pi = 3.14159265358979323846;
static constexpr Float InvPi = 0.31830988618379067154;
static constexpr Float Inv2Pi = 0.15915494309189533577;
static constexpr Float Inv4Pi = 0.07957747154594766788;
static constexpr Float PiOver2 = 1.57079632679489661923;
static constexpr Float PiOver4 = 0.78539816339744830961;
static constexpr Float Sqrt2 = 1.41421356237309504880;


// Assertions
#include <cassert>
#define CHECK(val1) assert(val1)
#define CHECK_EQ(val1,val2) CHECK(val1==val2)
#define CHECK_NE(val1, val2) CHECK(val1!=val2)
#define CHECK_LE(val1, val2) CHECK(val1<= val2)
#define CHECK_LT(val1,val2) CHECK(val1<val2)
#define CHECK_GE(val1, val2) CHECK(val1>= val2)
#define CHECK_GT(val1,val2) CHECK(val1>val2)


#define DCHECK(condition) CHECK(condition)
#define DCHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) CHECK_GT(val1, val2)

// Common defines
#ifndef L1_CACHE_LINE_SIZE
#define L1_CACHE_LINE_SIZE 64
#endif


// Global Inline Functions
inline uint32_t FloatToBits(float f) {
	uint32_t ui;
	memcpy(&ui, &f, sizeof(float));
	return ui;
}

inline float BitsToFloat(uint32_t ui) {
	float f;
	memcpy(&f, &ui, sizeof(uint32_t));
	return f;
}

inline uint64_t FloatToBits(double f) {
	uint64_t ui;
	memcpy(&ui, &f, sizeof(double));
	return ui;
}

inline double BitsToFloat(uint64_t ui) {
	double f;
	memcpy(&f, &ui, sizeof(uint64_t));
	return f;
}

inline float NextFloatUp(float v) {
	// Handle infinity and negative zero for _NextFloatUp()_
	if (std::isinf(v) && v > 0.) return v;
	if (v == -0.f) v = 0.f;

	// Advance _v_ to next higher float
	uint32_t ui = FloatToBits(v);
	if (v >= 0)
		++ui;
	else
		--ui;
	return BitsToFloat(ui);
}

inline float NextFloatDown(float v) {
	// Handle infinity and positive zero for _NextFloatDown()_
	if (std::isinf(v) && v < 0.) return v;
	if (v == 0.f) v = -0.f;
	uint32_t ui = FloatToBits(v);
	if (v > 0)
		--ui;
	else
		++ui;
	return BitsToFloat(ui);
}

inline Float gamma(int n) {
	return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

inline Float GammaCorrect(Float value) {
	if (value <= 0.0031308f) return 12.92f * value;
	return 1.055f * std::pow(value, (Float)(1.f / 2.4f)) - 0.055f;
}

inline Float InverseGammaCorrect(Float value) {
	if (value <= 0.04045f) return value * 1.f / 12.92f;
	return std::pow((value + 0.055f) * 1.f / 1.055f, (Float)2.4f);
}

template <typename T, typename U, typename V>
inline T Clamp(T val, U low, V high) {
	if (val < low)
		return low;
	else if (val > high)
		return high;
	else
		return val;
}

template <typename T>
inline T Mod(T a, T b) {
	T result = a - (a / b) * b;
	return (T)((result < 0) ? result + b : result);
}

template <>
inline Float Mod(Float a, Float b) {
	return std::fmod(a, b);
}

inline Float Radians(Float deg) { return (Pi / 180) * deg; }

inline Float Degrees(Float rad) { return (180 / Pi) * rad; }

template <typename T>
inline constexpr bool IsPowerOf2(T v) {
	return v && !(v & (v - 1));
}

inline void Error(const char* format, ...) {
	va_list args;
	va_start(args, format);
	printf(format, args);
	va_end(args);
}
