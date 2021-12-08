#include "benchmark/benchmark.h"
#include "CudaBenchmarkUtil.cuh"
#include <chrono>

// Nasty hack such that we can benchmark private functions
#define private public
#include "Scene.h"
#undef private

NAMESPACE_DPHPC_BEGIN


std::vector<std::string> SceneNames = {
    "conference",
    "fairyforest",
    "sibenik",
    "sanmiguel"
};

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


template <int SceneIndex> static void BM_CudaBVHBuilder(benchmark::State& state) {
    
    Scene* scene;
    const std::string SceneName = SceneNames[SceneIndex];
    const std::string ScenePath = "Scenes/" + SceneName + "/" + SceneName + ".obj";

    // WarmUp GPU
    WarmUpGPU();

    for (auto _ : state) {
       
        scene = new Scene();
        scene->LoadMeshFromFile(ScenePath);

        CUDA_SYNC_CHECK();
        auto start = std::chrono::high_resolution_clock::now();
        
        scene->BuildBVH(BVHBuilderType::CudaBVHBuilder);
        
        CUDA_SYNC_CHECK();
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 0)->Name("BM_CudaBVHBuilder/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 1)->Name("BM_CudaBVHBuilder/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 2)->Name("BM_CudaBVHBuilder/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 3)->Name("BM_CudaBVHBuilder/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(true)->UseManualTime();


BENCHMARK_MAIN();

NAMESPACE_DPHPC_END
