#include "benchmark/benchmark.h"
#include "CudaBenchmarkUtil.cuh"
#include <chrono>

// Nasty hack such that we can benchmark private functions
#define private public
#include "Scene.h"
#include "CudaBVHBuilder.h"
#undef private

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

BENCHMARK_TEMPLATE(BM_RecursiveBVHBuilder, 0)->Name("BM_RecursiveBVHBuilder/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_RecursiveBVHBuilder, 1)->Name("BM_RecursiveBVHBuilder/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_RecursiveBVHBuilder, 2)->Name("BM_RecursiveBVHBuilder/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_RecursiveBVHBuilder, 3)->Name("BM_RecursiveBVHBuilder/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(true);

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

        CUDA_SYNC_CHECK();
        state.ResumeTiming();
        auto start = std::chrono::high_resolution_clock::now();
        
        scene->BuildBVH(BVHBuilderType::CudaBVHBuilder);
        
        CUDA_SYNC_CHECK();
        auto end = std::chrono::high_resolution_clock::now();
        state.PauseTiming();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
        state.ResumeTiming();
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 0)->Name("BM_CudaBVHBuilder/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 1)->Name("BM_CudaBVHBuilder/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 2)->Name("BM_CudaBVHBuilder/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 3)->Name("BM_CudaBVHBuilder/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime();

/**
 * This function benchmarks the CudaBVHBuilder. Loading the mesh from the file is excluded.
 */
template <int SceneIndex> static void BM_CudaBVHBuilder_GenerateMortonCodes(benchmark::State& state) {
    
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

        const unsigned int nPrimitives = builder->primitiveInfo.size();
	    BVHPrimitiveInfoWithIndex* dPrimitiveInfo = builder->PrepareDevicePrimitiveInfo(nPrimitives);
        unsigned int* dMortonCodes;
	    unsigned int* dMortonIndices;

        // Start measure
        CUDA_SYNC_CHECK();
        state.ResumeTiming();
        auto start = std::chrono::high_resolution_clock::now();
        
	    builder->GenerateMortonCodesHelper(dPrimitiveInfo, &dMortonCodes, &dMortonIndices, nPrimitives);
        
        CUDA_SYNC_CHECK();
        auto end = std::chrono::high_resolution_clock::now();
        state.PauseTiming();
        // End measure

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());

        // Clean Up
        cudaFree(dMortonCodes);
        cudaFree(dMortonIndices);
        cudaFree(dPrimitiveInfo);
        delete builder;

        state.ResumeTiming();
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_GenerateMortonCodes, 0)->Name("BM_CudaBVHBuilder_GenerateMortonCodes/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_GenerateMortonCodes, 1)->Name("BM_CudaBVHBuilder_GenerateMortonCodes/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_GenerateMortonCodes, 2)->Name("BM_CudaBVHBuilder_GenerateMortonCodes/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_GenerateMortonCodes, 3)->Name("BM_CudaBVHBuilder_GenerateMortonCodes/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime();

BENCHMARK_MAIN();

NAMESPACE_DPHPC_END
