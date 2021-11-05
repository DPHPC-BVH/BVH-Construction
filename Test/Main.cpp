#include "../Source/Scene.h"
#include "../Source/Renderer.h"
#include "../Source/Parallel.h"

using namespace DPHPC;



int main(int argc, char** argv) {
	ParallelInit();

	std::string meshes[4] = {
		"Scenes/conference/conference.obj",
		"Scenes/fairyforest/fairyforest.obj",
		"Scenes/sibenik/sibenik.obj",
		"Scenes/sanmiguel/sanmiguel.obj"
	};

	Renderer::RenderOption options[4] = {
		{Point2i(1600, 900), 64, 15.0f},
		{Point2i(1600, 900), 32, 3.0f},
		{Point2i(1600, 900), 32, 10.0f},
		{Point2i(1600, 900), 32, 1.5f}
	};

	std::string CameraSignatures[4] = {
		"Y1BR00IkZd/0aA9X/0/Gy8Px1ca7Tw19///c/05frY109Qx7w////m100",
		"mF5Gz1SuO1z/ZMooz11Q0bGz/CCNxx18///m007toC10AnAHx///Uy200",
		"steO/0TlN1z1tsDg/03InaMz/bqZxx/9///c/05frY109Qx7w////m100",
		"Yciwz1oRQmz/Xvsm005CwjHx/b70nx18tVI7005frY108Y/:x/v3/z100"
	};

	int idx = 0;

	Scene scene;
	scene.LoadMesh(meshes[idx]);
	
	Renderer renderer(scene, options[idx]);
	renderer.LoadCameraSignature(CameraSignatures[idx]);

	renderer.Render();
	
	renderer.WriteImage("Scenes/result.pfm");

	ParallelCleanup();
}