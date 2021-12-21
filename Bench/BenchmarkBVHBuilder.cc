#include "benchmark/benchmark.h"
#include "CudaBenchmarkUtil.cuh"
#include <chrono>
#include "Timer.h"

// Nasty hack such that we can benchmark private functions
//#define private public
#include "Scene.h"
#include "CudaBVHBuilder.h"
//#undef private

NAMESPACE_DPHPC_BEGIN


std::vector<std::string> SceneNames = {
    "conference",
    "fairyforest",
    "sibenik",
    "sanmiguel"
};

/**
 * This function benchmarks the RecursiveBVHBuilder. Loading the mesh from the file is excluded.
 */
template <int SceneIndex> static void BM_RecursiveBVHBuilder(benchmark::State& state) {
    
    Scene* scene;
    const std::string SceneName = SceneNames[SceneIndex];
    const std::string ScenePath = "Scenes/" + SceneName + "/" + SceneName + ".obj";

    for (auto _ : state) {
        state.PauseTiming();
        scene = new Scene();
        scene->LoadMeshFromFile(ScenePath);
        state.ResumeTiming();

        scene->BuildBVH(BVHBuilderType::RecursiveBVHBuilder);

    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

//BENCHMARK_TEMPLATE(BM_RecursiveBVHBuilder, 0)->Name("BM_RecursiveBVHBuilder/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(false);
//BENCHMARK_TEMPLATE(BM_RecursiveBVHBuilder, 1)->Name("BM_RecursiveBVHBuilder/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(false);
//BENCHMARK_TEMPLATE(BM_RecursiveBVHBuilder, 2)->Name("BM_RecursiveBVHBuilder/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(false);
//BENCHMARK_TEMPLATE(BM_RecursiveBVHBuilder, 3)->Name("BM_RecursiveBVHBuilder/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(false);

/**
 * This function benchmarks the CudaBVHBuilder. Loading the mesh from the file is excluded.
 */
template <int SceneIndex> static void BM_CudaBVHBuilder(benchmark::State& state) {
    
    Scene* scene;
    const std::string SceneName = SceneNames[SceneIndex];
    const std::string ScenePath = "Scenes/" + SceneName + "/" + SceneName + ".obj";

    // WarmUp GPU
    WarmUpGPU();

    for (auto _ : state) {
        
        state.PauseTiming();
        scene = new Scene();
        scene->LoadMeshFromFile(ScenePath);

        // Additional GPU Timer
        TimerGPU timer;

        // Start Timers
        state.ResumeTiming();
        timer.Start();

        scene->BuildBVH(BVHBuilderType::CudaBVHBuilder);
        
        // End Timers
        double elapsed_microseconds = timer.Stop();
        state.PauseTiming();
        
        state.SetIterationTime(elapsed_microseconds / 1e6);
        state.ResumeTiming();
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 0)->Name("BM_CudaBVHBuilder/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 1)->Name("BM_CudaBVHBuilder/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 2)->Name("BM_CudaBVHBuilder/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 3)->Name("BM_CudaBVHBuilder/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();

/**
 * This function benchmarks the code that computes the bounding boxes in CudaBVHBuilder.
 */
template <int SceneIndex, int blockNum = -1> static void BM_CudaBVHBuilderAlgorithmOnly(benchmark::State& state) {
    
    Scene* scene;
    const std::string SceneName = SceneNames[SceneIndex];
    const std::string ScenePath = "Scenes/" + SceneName + "/" + SceneName + ".obj";

    // WarmUp GPU
    WarmUpGPU();

    for (auto _ : state) {
        
        state.PauseTiming();
        scene = new Scene();
        // Load mesh from file
        scene->LoadMeshFromFile(ScenePath);

        // Prepare data for BVH construction
        std::vector<std::shared_ptr<Primitive>> pTriangles;
	    pTriangles.reserve(scene->numTriangles);
	    for (int i = 0; i < scene->numTriangles; ++i) {
		    pTriangles.push_back(std::make_shared<Triangle>(scene->triangles[i]));
	    }
        scene->bvh = BVH(pTriangles);
        CudaBVHBuilder* builder = new CudaBVHBuilder(scene->bvh);

        builder->AllocAuxBuffers();
        builder->SetBlockNum(blockNum);


        // Additional GPU Timer
        TimerGPU timer;

        // Start Timers
        state.ResumeTiming();
        timer.Start();
        
        // 1. Compute Morton Codes
        builder->GenerateMortonCodesHelper();

        // 2. Sort Morton Codes
	    builder->SortMortonCodesHelper();

        // 3. Build tree hierarchy of CudaBVHBuildNodes
       builder->BuildTreeHierarchyHelper();

        // 4. Compute Bounding Boxes of each node
        builder->ComputeBoundingBoxesHelper();

        // End Timers
        double elapsed_microseconds = timer.Stop();
        state.PauseTiming();
        
        // 5. Flatten Tree and order BVH::primitives according to dMortonIndicesSorted
        builder->PermutePrimitivesAndFlattenTree();
        
        state.SetIterationTime(elapsed_microseconds / 1e6);

        // Clean Up
        delete builder;

        state.ResumeTiming();
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

//BENCHMARK_TEMPLATE(BM_CudaBVHBuilderAlgorithmOnly, 0)->Name("BM_CudaBVHBuilderAlgorithmOnly/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilderAlgorithmOnly, 1)->Name("BM_CudaBVHBuilderAlgorithmOnly/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilderAlgorithmOnly, 2)->Name("BM_CudaBVHBuilderAlgorithmOnly/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilderAlgorithmOnly, 3)->Name("BM_CudaBVHBuilderAlgorithmOnly/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();



/**
 * This function benchmarks the code to generate morton code in CudaBVHBuilder. Loading the mesh from the file is excluded.
 */
template <int SceneIndex, int blockNum = -1> static void BM_CudaBVHBuilder_GenerateMortonCodes(benchmark::State& state) {
    
    Scene* scene;
    const std::string SceneName = SceneNames[SceneIndex];
    const std::string ScenePath = "Scenes/" + SceneName + "/" + SceneName + ".obj";

    // WarmUp GPU
    WarmUpGPU();

    for (auto _ : state) {
        
        state.PauseTiming();
        scene = new Scene();
        // Load mesh from file
        scene->LoadMeshFromFile(ScenePath);

        // Prepare data for BVH construction
        std::vector<std::shared_ptr<Primitive>> pTriangles;
	    pTriangles.reserve(scene->numTriangles);
	    for (int i = 0; i < scene->numTriangles; ++i) {
		    pTriangles.push_back(std::make_shared<Triangle>(scene->triangles[i]));
	    }
        scene->bvh = BVH(pTriangles);
        CudaBVHBuilder* builder = new CudaBVHBuilder(scene->bvh);

        builder->AllocAuxBuffers();
        builder->SetBlockNum(blockNum);

        // Additional GPU Timer
        TimerGPU timer;

        // Start Timers
        state.ResumeTiming();
        timer.Start();
        
        // 1. Compute Morton Codes
	    builder->GenerateMortonCodesHelper();
        
        // End Timers
        double elapsed_microseconds = timer.Stop();
        state.PauseTiming();
        
        state.SetIterationTime(elapsed_microseconds / 1e6);

        // Clean Up
        delete builder;

        state.ResumeTiming();
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_GenerateMortonCodes, 0)->Name("BM_CudaBVHBuilder_GenerateMortonCodes/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_GenerateMortonCodes, 1)->Name("BM_CudaBVHBuilder_GenerateMortonCodes/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_GenerateMortonCodes, 2)->Name("BM_CudaBVHBuilder_GenerateMortonCodes/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_GenerateMortonCodes, 3)->Name("BM_CudaBVHBuilder_GenerateMortonCodes/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();

/**
 * This function benchmarks the code to sort morton code in CudaBVHBuilder.
 */
template <int SceneIndex> static void BM_CudaBVHBuilder_SortMortonCodes(benchmark::State& state) {
    
    Scene* scene;
    const std::string SceneName = SceneNames[SceneIndex];
    const std::string ScenePath = "Scenes/" + SceneName + "/" + SceneName + ".obj";

    // WarmUp GPU
    WarmUpGPU();

    for (auto _ : state) {
        
        state.PauseTiming();
        scene = new Scene();
        // Load mesh from file
        scene->LoadMeshFromFile(ScenePath);

        // Prepare data for BVH construction
        std::vector<std::shared_ptr<Primitive>> pTriangles;
	    pTriangles.reserve(scene->numTriangles);
	    for (int i = 0; i < scene->numTriangles; ++i) {
		    pTriangles.push_back(std::make_shared<Triangle>(scene->triangles[i]));
	    }
        scene->bvh = BVH(pTriangles);
        CudaBVHBuilder* builder = new CudaBVHBuilder(scene->bvh);

        builder->AllocAuxBuffers();
        
        // 1. Compute Morton Codes
        builder->GenerateMortonCodesHelper();

        // Additional GPU Timer
        TimerGPU timer;

        // Start Timers
        state.ResumeTiming();
        timer.Start();
        
        // 2. Sort Morton Codes
	    builder->SortMortonCodesHelper();
        
        // End Timers
        double elapsed_microseconds = timer.Stop();
        state.PauseTiming();
        
        state.SetIterationTime(elapsed_microseconds / 1e6);

        // Clean Up
        delete builder;

        state.ResumeTiming();
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_SortMortonCodes, 0)->Name("BM_CudaBVHBuilder_SortMortonCodes/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_SortMortonCodes, 1)->Name("BM_CudaBVHBuilder_SortMortonCodes/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_SortMortonCodes, 2)->Name("BM_CudaBVHBuilder_SortMortonCodes/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_SortMortonCodes, 3)->Name("BM_CudaBVHBuilder_SortMortonCodes/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();

/**
 * This function benchmarks the code to build the three hierarchy in CudaBVHBuilder.
 */
template <int SceneIndex, int blockNum = -1> static void BM_CudaBVHBuilder_BuildTreeHierarchy(benchmark::State& state) {
    
    Scene* scene;
    const std::string SceneName = SceneNames[SceneIndex];
    const std::string ScenePath = "Scenes/" + SceneName + "/" + SceneName + ".obj";

    // WarmUp GPU
    WarmUpGPU();

    for (auto _ : state) {
        
        state.PauseTiming();
        scene = new Scene();
        // Load mesh from file
        scene->LoadMeshFromFile(ScenePath);

        // Prepare data for BVH construction
        std::vector<std::shared_ptr<Primitive>> pTriangles;
	    pTriangles.reserve(scene->numTriangles);
	    for (int i = 0; i < scene->numTriangles; ++i) {
		    pTriangles.push_back(std::make_shared<Triangle>(scene->triangles[i]));
	    }
        scene->bvh = BVH(pTriangles);
        CudaBVHBuilder* builder = new CudaBVHBuilder(scene->bvh);

        builder->AllocAuxBuffers();
        builder->SetBlockNum(blockNum);
        
        // 1. Compute Morton Codes
        builder->GenerateMortonCodesHelper();

        // 2. Sort Morton Codes
	    builder->SortMortonCodesHelper();

        // Additional GPU Timer
        TimerGPU timer;

        // Start Timers
        state.ResumeTiming();
        timer.Start();
        
        // 3. Build tree hierarchy of CudaBVHBuildNodes
        builder->BuildTreeHierarchyHelper();
       
        // End Timers
        double elapsed_microseconds = timer.Stop();
        state.PauseTiming();
        
        state.SetIterationTime(elapsed_microseconds / 1e6);

        // Clean Up
        delete builder;

        state.ResumeTiming();
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_BuildTreeHierarchy, 0)->Name("BM_CudaBVHBuilder_BuildTreeHierarchy/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_BuildTreeHierarchy, 1)->Name("BM_CudaBVHBuilder_BuildTreeHierarchy/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_BuildTreeHierarchy, 2)->Name("BM_CudaBVHBuilder_BuildTreeHierarchy/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_BuildTreeHierarchy, 3)->Name("BM_CudaBVHBuilder_BuildTreeHierarchy/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();


/**
 * This function benchmarks the code that computes the bounding boxes in CudaBVHBuilder.
 */
template <int SceneIndex, int blockNum = -1> static void BM_CudaBVHBuilder_ComputeBoundingBoxes(benchmark::State& state) {
    
    Scene* scene;
    const std::string SceneName = SceneNames[SceneIndex];
    const std::string ScenePath = "Scenes/" + SceneName + "/" + SceneName + ".obj";

    // WarmUp GPU
    WarmUpGPU();

    for (auto _ : state) {
        
        state.PauseTiming();
        scene = new Scene();
        // Load mesh from file
        scene->LoadMeshFromFile(ScenePath);

        // Prepare data for BVH construction
        std::vector<std::shared_ptr<Primitive>> pTriangles;
	    pTriangles.reserve(scene->numTriangles);
	    for (int i = 0; i < scene->numTriangles; ++i) {
		    pTriangles.push_back(std::make_shared<Triangle>(scene->triangles[i]));
	    }
        scene->bvh = BVH(pTriangles);
        CudaBVHBuilder* builder = new CudaBVHBuilder(scene->bvh);

        builder->AllocAuxBuffers();
        builder->SetBlockNum(blockNum);
        
        // 1. Compute Morton Codes
        builder->GenerateMortonCodesHelper();

        // 2. Sort Morton Codes
	    builder->SortMortonCodesHelper();

        // 3. Build tree hierarchy of CudaBVHBuildNodes
        builder->BuildTreeHierarchyHelper();

        // Additional GPU Timer
        TimerGPU timer;

        // Start Timers
        state.ResumeTiming();
        timer.Start();
        
        // 4. Compute Bounding Boxes of each node
	    builder->ComputeBoundingBoxesHelper();

        // End Timers
        double elapsed_microseconds = timer.Stop();
        state.PauseTiming();
        
        state.SetIterationTime(elapsed_microseconds / 1e6);

        // Clean Up
        delete builder;

        state.ResumeTiming();
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_ComputeBoundingBoxes, 0, false)->Name("BM_CudaBVHBuilder_ComputeBoundingBoxes/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_ComputeBoundingBoxes, 1, false)->Name("BM_CudaBVHBuilder_ComputeBoundingBoxes/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_ComputeBoundingBoxes, 2, false)->Name("BM_CudaBVHBuilder_ComputeBoundingBoxes/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_ComputeBoundingBoxes, 3, false)->Name("BM_CudaBVHBuilder_ComputeBoundingBoxes/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_ComputeBoundingBoxes, 0, true)->Name("BM_CudaBVHBuilder_ComputeBoundingBoxesWithSharedMemory/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_ComputeBoundingBoxes, 1, true)->Name("BM_CudaBVHBuilder_ComputeBoundingBoxesWithSharedMemory/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_ComputeBoundingBoxes, 2, true)->Name("BM_CudaBVHBuilder_ComputeBoundingBoxesWithSharedMemory/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_ComputeBoundingBoxes, 3, true)->Name("BM_CudaBVHBuilder_ComputeBoundingBoxesWithSharedMemory/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();


/**
 * This function benchmarks the code that computes the bounding boxes in CudaBVHBuilder.
 */
template <int SceneIndex> static void BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree(benchmark::State& state) {
    
    Scene* scene;
    const std::string SceneName = SceneNames[SceneIndex];
    const std::string ScenePath = "Scenes/" + SceneName + "/" + SceneName + ".obj";

    // WarmUp GPU
    WarmUpGPU();

    for (auto _ : state) {
        
        state.PauseTiming();
        scene = new Scene();
        // Load mesh from file
        scene->LoadMeshFromFile(ScenePath);

        // Prepare data for BVH construction
        std::vector<std::shared_ptr<Primitive>> pTriangles;
	    pTriangles.reserve(scene->numTriangles);
	    for (int i = 0; i < scene->numTriangles; ++i) {
		    pTriangles.push_back(std::make_shared<Triangle>(scene->triangles[i]));
	    }
        scene->bvh = BVH(pTriangles);
        CudaBVHBuilder* builder = new CudaBVHBuilder(scene->bvh);

        builder->AllocAuxBuffers();
        
        // 1. Compute Morton Codes
        builder->GenerateMortonCodesHelper();

        // 2. Sort Morton Codes
	    builder->SortMortonCodesHelper();

        // 3. Build tree hierarchy of CudaBVHBuildNodes
        builder->BuildTreeHierarchyHelper();

        // 4. Compute Bounding Boxes of each node
        builder->ComputeBoundingBoxesHelper();

        // Additional GPU Timer
        TimerGPU timer;

        // Start Timers
        state.ResumeTiming();
        timer.Start();
        
        // 5. Flatten Tree and order BVH::primitives according to dMortonIndicesSorted
        builder->PermutePrimitivesAndFlattenTree();
       
        // End Timers
        double elapsed_microseconds = timer.Stop();
        state.PauseTiming();
        
        state.SetIterationTime(elapsed_microseconds / 1e6);

        // Clean Up
        delete builder;

        state.ResumeTiming();
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree, 0)->Name("BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree, 1)->Name("BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree, 2)->Name("BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();
//BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree, 3)->Name("BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(false)->UseManualTime();

#define IMPLEMENT_SCALABILITY_TEST(SceneIndex, func) \
    BENCHMARK_TEMPLATE(func, 0, 1)->Name(#func + SceneNames[SceneIndex] + "/1")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 2)->Name(#func + SceneNames[SceneIndex] + "/2")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 3)->Name(#func + SceneNames[SceneIndex] + "/3")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 4)->Name(#func + SceneNames[SceneIndex] + "/4")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 5)->Name(#func + SceneNames[SceneIndex] + "/5")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 6)->Name(#func + SceneNames[SceneIndex] + "/6")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 7)->Name(#func + SceneNames[SceneIndex] + "/7")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 8)->Name(#func + SceneNames[SceneIndex] + "/8")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 9)->Name(#func + SceneNames[SceneIndex] + "/9")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 10)->Name(#func + SceneNames[SceneIndex] + "/10")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 11)->Name(#func + SceneNames[SceneIndex] + "/11")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 12)->Name(#func + SceneNames[SceneIndex] + "/12")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 13)->Name(#func + SceneNames[SceneIndex] + "/13")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 14)->Name(#func + SceneNames[SceneIndex] + "/14")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 15)->Name(#func + SceneNames[SceneIndex] + "/15")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 16)->Name(#func + SceneNames[SceneIndex] + "/16")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 17)->Name(#func + SceneNames[SceneIndex] + "/17")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 18)->Name(#func + SceneNames[SceneIndex] + "/18")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 19)->Name(#func + SceneNames[SceneIndex] + "/19")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 20)->Name(#func + SceneNames[SceneIndex] + "/20")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 21)->Name(#func + SceneNames[SceneIndex] + "/21")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 22)->Name(#func + SceneNames[SceneIndex] + "/22")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 23)->Name(#func + SceneNames[SceneIndex] + "/23")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 24)->Name(#func + SceneNames[SceneIndex] + "/24")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 25)->Name(#func + SceneNames[SceneIndex] + "/25")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 26)->Name(#func + SceneNames[SceneIndex] + "/26")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 27)->Name(#func + SceneNames[SceneIndex] + "/27")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 28)->Name(#func + SceneNames[SceneIndex] + "/28")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 29)->Name(#func + SceneNames[SceneIndex] + "/29")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 30)->Name(#func + SceneNames[SceneIndex] + "/30")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 32)->Name(#func + SceneNames[SceneIndex] + "/32")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 34)->Name(#func + SceneNames[SceneIndex] + "/34")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 36)->Name(#func + SceneNames[SceneIndex] + "/36")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 38)->Name(#func + SceneNames[SceneIndex] + "/38")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 40)->Name(#func + SceneNames[SceneIndex] + "/40")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 45)->Name(#func + SceneNames[SceneIndex] + "/45")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 50)->Name(#func + SceneNames[SceneIndex] + "/50")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 60)->Name(#func + SceneNames[SceneIndex] + "/60")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 70)->Name(#func + SceneNames[SceneIndex] + "/70")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 80)->Name(#func + SceneNames[SceneIndex] + "/80")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 90)->Name(#func + SceneNames[SceneIndex] + "/90")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 100)->Name(#func + SceneNames[SceneIndex] + "/100")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 125)->Name(#func + SceneNames[SceneIndex] + "/125")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 150)->Name(#func + SceneNames[SceneIndex] + "/150")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 175)->Name(#func + SceneNames[SceneIndex] + "/175")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 200)->Name(#func + SceneNames[SceneIndex] + "/200")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 250)->Name(#func + SceneNames[SceneIndex] + "/250")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 300)->Name(#func + SceneNames[SceneIndex] + "/300")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 350)->Name(#func + SceneNames[SceneIndex] + "/350")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, 400)->Name(#func + SceneNames[SceneIndex] + "/400")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \
	BENCHMARK_TEMPLATE(func, 0, -1)->Name(#func + SceneNames[SceneIndex] + "/-1")->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime(); \


IMPLEMENT_SCALABILITY_TEST(0, BM_CudaBVHBuilder_GenerateMortonCodes)
IMPLEMENT_SCALABILITY_TEST(0, BM_CudaBVHBuilder_BuildTreeHierarchy)
IMPLEMENT_SCALABILITY_TEST(0, BM_CudaBVHBuilder_ComputeBoundingBoxes)
IMPLEMENT_SCALABILITY_TEST(0, BM_CudaBVHBuilderAlgorithmOnly)

IMPLEMENT_SCALABILITY_TEST(1, BM_CudaBVHBuilder_GenerateMortonCodes)
IMPLEMENT_SCALABILITY_TEST(1, BM_CudaBVHBuilder_BuildTreeHierarchy)
IMPLEMENT_SCALABILITY_TEST(1, BM_CudaBVHBuilder_ComputeBoundingBoxes)
IMPLEMENT_SCALABILITY_TEST(1, BM_CudaBVHBuilderAlgorithmOnly)

IMPLEMENT_SCALABILITY_TEST(2, BM_CudaBVHBuilder_GenerateMortonCodes)
IMPLEMENT_SCALABILITY_TEST(2, BM_CudaBVHBuilder_BuildTreeHierarchy)
IMPLEMENT_SCALABILITY_TEST(2, BM_CudaBVHBuilder_ComputeBoundingBoxes)
IMPLEMENT_SCALABILITY_TEST(2, BM_CudaBVHBuilderAlgorithmOnly)

IMPLEMENT_SCALABILITY_TEST(3, BM_CudaBVHBuilder_GenerateMortonCodes)
IMPLEMENT_SCALABILITY_TEST(3, BM_CudaBVHBuilder_BuildTreeHierarchy)
IMPLEMENT_SCALABILITY_TEST(3, BM_CudaBVHBuilder_ComputeBoundingBoxes)
IMPLEMENT_SCALABILITY_TEST(3, BM_CudaBVHBuilderAlgorithmOnly)
BENCHMARK_MAIN();

NAMESPACE_DPHPC_END
