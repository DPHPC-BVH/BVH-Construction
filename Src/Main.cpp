#include "Scene.h"
#include "Renderer.h"
#include "Parallel.h"
#include "Argparse.h"

using namespace DPHPC;



int main(int argc, char** argv) {

	argparse::ArgumentParser program("main");

	program.add_argument("-s", "--scene")
  	.help("Choose the scene, conference (0), fairyforest (1), sibenik (2), sanmiguel (3)")
	.scan<'i', int>()
  	.default_value(0);
	
	program.add_argument("--construction-only")
  	.help("If enabled no rendering is done")
  	.default_value(false)
	.implicit_value(true);


	static const std::map<std::string, BVHBuilderType> BVHConstructionTypes = { 
			{"recursive", BVHBuilderType::RecursiveBVHBuilder},
			{"gpu", BVHBuilderType::CudaBVHBuilder}
		};
	program.add_argument("--method")
  	.help("Chooses BVH constructions methods. Possible choices: recursive, gpu.")
  	.default_value(BVHBuilderType::RecursiveBVHBuilder)
	.action([](const std::string& value) {
		auto search = BVHConstructionTypes.find(value);
    	if (search != BVHConstructionTypes.end()) {
      		return search->second;
    	}
    	return BVHBuilderType::RecursiveBVHBuilder;
  	});

	try {
  		program.parse_args(argc, argv);
	}	
	catch (const std::runtime_error& err) {
  		std::cerr << err.what() << std::endl;
  		std::cerr << program;
  		std::exit(1);
	}



	ParallelInit();

	std::string meshes[4] = {
		"Scenes/conference/conference.obj",
		"Scenes/fairyforest/fairyforest.obj",
		"Scenes/sibenik/sibenik.obj",
		"Scenes/sanmiguel/sanmiguel.obj"
	};

	Renderer::RenderOption options[4] = {
		{Point2i(800, 600), 32, 15.0f},
		{Point2i(800, 600), 32, 3.0f},
		{Point2i(800, 600), 32, 10.0f},
		{Point2i(800, 600), 32, 1.5f}
	};

	std::string CameraSignatures[4] = {
		"Y1BR00IkZd/0aA9X/0/Gy8Px1ca7Tw19///c/05frY109Qx7w////m100",
		"mF5Gz1SuO1z/ZMooz11Q0bGz/CCNxx18///m007toC10AnAHx///Uy200",
		"steO/0TlN1z1tsDg/03InaMz/bqZxx/9///c/05frY109Qx7w////m100",
		"Yciwz1oRQmz/Xvsm005CwjHx/b70nx18tVI7005frY108Y/:x/v3/z100"
	};

	int idx = program.get<int>("--scene");
	BVHBuilderType type = program.get<BVHBuilderType>("--method");



	Scene scene;
	scene.LoadMesh(meshes[idx], type);

	if(program.get<bool>("--construction-only") == false) {
		Renderer renderer(scene, options[idx]);
		renderer.LoadCameraSignature(CameraSignatures[idx]);

		renderer.Render();
	
		renderer.WriteImage("Scenes/result.pfm");
	}
	

	ParallelCleanup();
}