#include "Renderer.h"
#include "Parallel.h"
#include "Sampling.h"

NAMESPACE_DPHPC_BEGIN

void EncodeBits(std::string& dst, uint32_t v) {
	CHECK(v < 64);
	int base = (v < 12) ? '/' : (v < 38) ? 'A' - 12 : 'a' - 38;
	dst += (char)(v + base);
}

//------------------------------------------------------------------------

uint32_t DecodeBits(const char*& src) {
	if (*src >= '/' && *src <= ':') return *src++ - '/';
	if (*src >= 'A' && *src <= 'Z') return *src++ - 'A' + 12;
	if (*src >= 'a' && *src <= 'z') return *src++ - 'a' + 38;
	CHECK(0); // "CameraControls: Invalid signature!");
	return 0;
}

//------------------------------------------------------------------------

void EncodeFloat(std::string& dst, Float v) {
	uint32_t bits = FloatToBits(v);
	for (int i = 0; i < 32; i += 6)
		EncodeBits(dst, (bits >> i) & 0x3F);
}

//------------------------------------------------------------------------

Float DecodeFloat(const char*& src) {
	uint32_t bits = 0;
	for (int i = 0; i < 32; i += 6)
		bits |= DecodeBits(src) << i;
	return BitsToFloat(bits);
}

//------------------------------------------------------------------------

void EncodeDirection(std::string& dst, const Vector3f& v) {
	Vector3f a(std::abs(v.x), std::abs(v.y), std::abs(v.z));
	int axis = (a.x >= std::max(a.y, a.z)) ? 0 : (a.y >= a.z) ? 1 : 2;

	Vector3f tuv;
	switch (axis) {
	case 0:  tuv = v; break;
	case 1:  tuv = Vector3f(v.y, v.z, v.x); break;
	default: tuv = Vector3f(v.z, v.x, v.y); break;
	}

	int face = axis | ((tuv.x >= 0.0f) ? 0 : 4);
	if (tuv.y == 0.0f && tuv.z == 0.0f) {
		EncodeBits(dst, face | 8);
		return;
	}

	EncodeBits(dst, face);
	EncodeFloat(dst, tuv.y / abs(tuv.x));
	EncodeFloat(dst, tuv.z / abs(tuv.x));
}

//------------------------------------------------------------------------

Vector3f DecodeDirection(const char*& src) {
	int face = DecodeBits(src);
	Vector3f tuv;
	tuv.x = ((face & 4) == 0) ? 1.0f : -1.0f;
	tuv.y = ((face & 8) == 0) ? DecodeFloat(src) : 0.0f;
	tuv.z = ((face & 8) == 0) ? DecodeFloat(src) : 0.0f;
	tuv = Normalize(tuv);

	switch (face & 3) {
	case 0:  return tuv;
	case 1:  return Vector3f(tuv.z, tuv.x, tuv.y);
	default: return Vector3f(tuv.y, tuv.z, tuv.x);
	}
}

Renderer::Renderer(const Scene& scene, RenderOption option)
	: scene(scene), option(option) {
	pixels.resize(option.resolution.x * option.resolution.y);
}



Renderer::~Renderer() {

}

void Renderer::LoadCameraSignature(const std::string& sig) {
	const char* src = sig.c_str();
	while (*src == ' ' || *src == '\t' || *src == '\n') src++;
	if (*src == '"') src++;

	Float   px = DecodeFloat(src);
	Float   py = DecodeFloat(src);
	Float   pz = DecodeFloat(src);
	Point3f origin = Point3f(px, py, pz);
	Vector3f forward = DecodeDirection(src);
	Vector3f up = DecodeDirection(src);
	Float   speed = DecodeFloat(src);  // Not used
	Float   fov = DecodeFloat(src);  // This seems to be the half fov...
	Float   znear = DecodeFloat(src);
	Float   zfar = DecodeFloat(src);
	bool  keepAligned = (DecodeBits(src) != 0);  // Not used

	if (*src == '"') src++;
	if (*src == ',') src++;
	while (*src == ' ' || *src == '\t' || *src == '\n') src++;
	CHECK(!*src); // "CameraControls: Invalid signature!");


	worldToView = LookAt(origin, origin + forward, up);
	viewToWorld = Transform(worldToView.GetInverseMatrix(), worldToView.GetMatrix());


	Matrix4x4 projMat;
	projMat.m[0][0] = 1.0f / std::tan(Radians(fov));
	projMat.m[1][1] = projMat.m[0][0] * option.resolution.x / option.resolution.y;
	projMat.m[2][2] = zfar / (zfar - znear);
	projMat.m[2][3] = 1.0f;
	projMat.m[3][2] = -znear * zfar / (zfar - znear);
	
	viewToClip = Transform(projMat);
	clipToView = Inverse(viewToClip);
}

void Renderer::Render() {
	const int tileSize = 16;
	Point2i nTiles((option.resolution.x + tileSize - 1) / tileSize, (option.resolution.y + tileSize - 1) / tileSize);
	// Each thread process one tile
	ParallelFor2D([&](Point2i tile) {
		// Compute sample bounds for tile
		int x0 = tile.x * tileSize;
		int x1 = std::min(x0 + tileSize, option.resolution.x);
		int y0 = tile.y * tileSize;
		int y1 = std::min(y0 + tileSize, option.resolution.y);
		Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));

		printf("Rendering tile %d,%d to %d,%d \n", x0, y0, x1, y1);



		// Loop over pixels in the tile
		for (Point2i pixel : tileBounds) {
			// Generate samples for this pixel 
			std::vector<Point2f> samples;
			samples.reserve(option.numAOSamples);
			for (int i = 0; i < option.numAOSamples; ++i) {
				//samples.push_back(Point2f(Float(i)/option.numAOSamples, RadicalInverse(1, i)));  // Hammersley sequence
				//samples.push_back(Point2f(RadicalInverse(1, i), RadicalInverse(2, i)));  // Halton sequence
				samples.push_back(Point2f(1.0f * std::rand() / RAND_MAX, 1.0f * std::rand() / RAND_MAX));
			}


			Float ao = 0.0f;
			Float xClip = Float(pixel.x) / option.resolution.x * 2.0f - 1.0f;
			Float yClip = Float(pixel.y) / option.resolution.y * 2.0f - 1.0f;  // May need to be flipped... Depends on the definition of coordinate system

			Point3f pixelClip = Point3f(xClip, yClip, 0);
			Point3f pixelView = clipToView(pixelClip);
			Ray primaryRay = viewToWorld(Ray(Point3f(0.0f, 0.0f, 0.0f), Normalize(Vector3f(pixelView))));

			Intersection primaryIntersection;
			if (scene.bvh.Intersect(primaryRay, &primaryIntersection)) {
				// Generate an arbitrary coordinate system
				Vector3f n = Vector3f(Faceforward(primaryIntersection.n, -primaryRay.d));
				Vector3f s, t;
				CoordinateSystem(n, &s, &t);
				
				for (int i = 0; i < option.numAOSamples; ++i) 		{
					const bool cosSample = false;
					Vector3f wi;
					Float pdf;
					if (cosSample) {
						wi = CosineSampleHemisphere(samples[i]);
						pdf = CosineHemispherePdf(std::abs(wi.z));
					}
					else {
						wi = UniformSampleHemisphere(samples[i]);
						pdf = UniformHemispherePdf();
					}

					// Transform wi from local frame to world space.
					wi = Vector3f(s.x * wi.x + t.x * wi.y + n.x * wi.z,
						s.y * wi.x + t.y * wi.y + n.y * wi.z,
						s.z * wi.x + t.z * wi.y + n.z * wi.z);

					const Float delta = 1e-5f;  // Slightly offset the ray origin

					if (!scene.bvh.IntersectP(Ray(primaryIntersection.p + wi * delta, wi, option.aoRadius)))
						ao += Dot(wi, n) / (pdf * option.numAOSamples);
				}

				////Output the normals for debugging
				//Normal3f p = primaryIntersection.n;
				//Pixel& pix = pixels[pixel.y * option.resolution.x + pixel.x];
				//pix.r = p.x * 0.5f + 0.5f;
				//pix.g = p.y * 0.5f + 0.5f;
				//pix.b = p.z * 0.5f + 0.5f;
			}

			Pixel& pix = pixels[pixel.y * option.resolution.x + pixel.x];
			pix.r = pix.g = pix.b = ao;

		}
	}, nTiles);
}

static bool WriteImagePFM(const std::string& filename, const Float* rgb,
	int width, int height) {
	FILE* fp;
	float scale;

	fp = fopen(filename.c_str(), "wb");
	if (!fp) {
		Error("Unable to open output PFM file \"%s\"", filename.c_str());
		return false;
	}

	std::unique_ptr<float[]> scanline(new float[3 * width]);

	// only write 3 channel PFMs here...
	if (fprintf(fp, "PF\n") < 0) goto fail;

	// write the width and height, which must be positive
	if (fprintf(fp, "%d %d\n", width, height) < 0) goto fail;

	// write the scale, which encodes endianness
	scale = -1.f;//hostLittleEndian ? -1.f : 1.f;
	if (fprintf(fp, "%f\n", scale) < 0) goto fail;

	// write the data from bottom left to upper right as specified by
	// http://netpbm.sourceforge.net/doc/pfm.html
	// The raster is a sequence of pixels, packed one after another, with no
	// delimiters of any kind. They are grouped by row, with the pixels in each
	// row ordered left to right and the rows ordered bottom to top.
	for (int y = height - 1; y >= 0; y--) {
		// in case Float is 'double', copy into a staging buffer that's
		// definitely a 32-bit float...
		for (int x = 0; x < 3 * width; ++x)
			scanline[x] = rgb[y * width * 3 + x];
		if (fwrite(&scanline[0], sizeof(float), width * 3, fp) <
			(size_t)(width * 3))
			goto fail;
	}

	fclose(fp);
	return true;

fail:
	Error("Error writing PFM file \"%s\"", filename.c_str());
	fclose(fp);
	return false;
}

void Renderer::WriteImage(std::string path) {
	std::unique_ptr<Float[]> rgb(new Float[3 * option.resolution.x * option.resolution.y]);
	Bounds2i pixelBound = Bounds2i(Point2i(), option.resolution);
	int offset = 0;
	for (Point2i pixel : pixelBound) 		{
		Pixel& p = pixels[pixel.y * option.resolution.x + pixel.x];
		rgb[3 * offset] = p.r;
		rgb[3 * offset + 1] = p.g;
		rgb[3 * offset + 2] = p.b;

		++offset;
	}

	WriteImagePFM(path, &rgb[0], option.resolution.x, option.resolution.y);
}

NAMESPACE_DPHPC_END



