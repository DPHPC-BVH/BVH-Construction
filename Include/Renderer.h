#pragma once
#include "DPHPC.h"
#include "Geometry.h"
#include "Transform.h"
#include "Scene.h"


NAMESPACE_DPHPC_BEGIN

// Ambient occlusion renderer
class Renderer { 
public:
	struct RenderOption {
		Point2i resolution;
		int numAOSamples;
		float aoRadius;
	};

	struct Pixel {
		Float r, g, b;
		Pixel() : r(0.0f), g(0.0f), b(0.0f)  {}
		Pixel(Float r, Float g, Float b) : r(r), g(g), b(b) {}
		Pixel(const Pixel& other) : r(other.r), g(other.g), b(other.b) {}
	};

	Renderer(const Scene& scene, RenderOption option);
	~Renderer();

	void LoadCameraSignature(const std::string& sig);

	void Render();
	void WriteImage(std::string path);

private:
	Transform worldToView, viewToWorld;  // world -> view space
	Transform viewToClip, clipToView;  // view -> clip space (projection). x,y,z in [-1,1]x[-1,1]x[0,1] after perspective divide

	const Scene& scene;
	RenderOption option;

	std::vector<Pixel> pixels;
};

NAMESPACE_DPHPC_END