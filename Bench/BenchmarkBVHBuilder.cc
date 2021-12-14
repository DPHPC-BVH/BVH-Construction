#include "benchmark/benchmark.h"
#include "CudaBenchmarkUtil.cuh"
#include <chrono>

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
        
        scene->BuildBVH(BVHBuilderType::CudaBVHBuilder);
        
        CUDA_SYNC_CHECK();
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 0)->Name("BM_CudaBVHBuilder/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 1)->Name("BM_CudaBVHBuilder/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 2)->Name("BM_CudaBVHBuilder/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder, 3)->Name("BM_CudaBVHBuilder/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(true);

/**
 * This function benchmarks the code to generate morton code in CudaBVHBuilder. Loading the mesh from the file is excluded.
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

        // Start measure
        CUDA_SYNC_CHECK();
        state.ResumeTiming();
        
        // 1. Compute Morton Codes
        unsigned int* dMortonCodes;
	    unsigned int* dMortonIndices;
	    builder->GenerateMortonCodesHelper(dPrimitiveInfo, &dMortonCodes, &dMortonIndices, nPrimitives);
        
        CUDA_SYNC_CHECK();
        state.PauseTiming();
        // End measure

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

BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_GenerateMortonCodes, 0)->Name("BM_CudaBVHBuilder_GenerateMortonCodes/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_GenerateMortonCodes, 1)->Name("BM_CudaBVHBuilder_GenerateMortonCodes/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_GenerateMortonCodes, 2)->Name("BM_CudaBVHBuilder_GenerateMortonCodes/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_GenerateMortonCodes, 3)->Name("BM_CudaBVHBuilder_GenerateMortonCodes/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(true);

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

        const unsigned int nPrimitives = builder->primitiveInfo.size();
	    BVHPrimitiveInfoWithIndex* dPrimitiveInfo = builder->PrepareDevicePrimitiveInfo(nPrimitives);
        
        // 1. Compute Morton Codes
        unsigned int* dMortonCodes;
	    unsigned int* dMortonIndices;
        builder->GenerateMortonCodesHelper(dPrimitiveInfo, &dMortonCodes, &dMortonIndices, nPrimitives);

        // Start measure
        CUDA_SYNC_CHECK();
        state.ResumeTiming();
        
        // 2. Sort Morton Codes
        unsigned int* dMortonCodesSorted;
	    unsigned int* dMortonIndicesSorted;
	    builder->SortMortonCodesHelper(dPrimitiveInfo, dMortonCodes, dMortonIndices, &dMortonCodesSorted, &dMortonIndicesSorted, nPrimitives);
        
        CUDA_SYNC_CHECK();
        state.PauseTiming();
        // End measure


        // Clean Up
        cudaFree(dMortonCodesSorted);
        cudaFree(dMortonIndicesSorted);
        cudaFree(dPrimitiveInfo);
        delete builder;

        state.ResumeTiming();
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_SortMortonCodes, 0)->Name("BM_CudaBVHBuilder_SortMortonCodes/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_SortMortonCodes, 1)->Name("BM_CudaBVHBuilder_SortMortonCodes/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_SortMortonCodes, 2)->Name("BM_CudaBVHBuilder_SortMortonCodes/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_SortMortonCodes, 3)->Name("BM_CudaBVHBuilder_SortMortonCodes/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(true);

/**
 * This function benchmarks the code to build the three hierarchy in CudaBVHBuilder.
 */
template <int SceneIndex> static void BM_CudaBVHBuilder_BuildTreeHierarchy(benchmark::State& state) {
    
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
        
        // 1. Compute Morton Codes
        unsigned int* dMortonCodes;
	    unsigned int* dMortonIndices;
        builder->GenerateMortonCodesHelper(dPrimitiveInfo, &dMortonCodes, &dMortonIndices, nPrimitives);

        // 2. Sort Morton Codes
        unsigned int* dMortonCodesSorted;
	    unsigned int* dMortonIndicesSorted;
	    builder->SortMortonCodesHelper(dPrimitiveInfo, dMortonCodes, dMortonIndices, &dMortonCodesSorted, &dMortonIndicesSorted, nPrimitives);

        // Start measure
        CUDA_SYNC_CHECK();
        state.ResumeTiming();
        
        // 3. Build tree hierarchy of CudaBVHBuildNodes
        CudaBVHBuildNode* dTree = builder->BuildTreeHierarchyHelper(dMortonCodesSorted, dMortonIndicesSorted, nPrimitives);
       
        CUDA_SYNC_CHECK();
        state.PauseTiming();
        // End measure

        // Clean Up
        cudaFree(dTree);
        cudaFree(dPrimitiveInfo);
        delete builder;

        state.ResumeTiming();
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_BuildTreeHierarchy, 0)->Name("BM_CudaBVHBuilder_BuildTreeHierarchy/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_BuildTreeHierarchy, 1)->Name("BM_CudaBVHBuilder_BuildTreeHierarchy/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_BuildTreeHierarchy, 2)->Name("BM_CudaBVHBuilder_BuildTreeHierarchy/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_BuildTreeHierarchy, 3)->Name("BM_CudaBVHBuilder_BuildTreeHierarchy/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(true);


/**
 * This function benchmarks the code that computes the bounding boxes in CudaBVHBuilder.
 */
template <int SceneIndex> static void BM_CudaBVHBuilder_ComputeBoundingBoxes(benchmark::State& state) {
    
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
        
        // 1. Compute Morton Codes
        unsigned int* dMortonCodes;
	    unsigned int* dMortonIndices;
        builder->GenerateMortonCodesHelper(dPrimitiveInfo, &dMortonCodes, &dMortonIndices, nPrimitives);

        // 2. Sort Morton Codes
        unsigned int* dMortonCodesSorted;
	    unsigned int* dMortonIndicesSorted;
	    builder->SortMortonCodesHelper(dPrimitiveInfo, dMortonCodes, dMortonIndices, &dMortonCodesSorted, &dMortonIndicesSorted, nPrimitives);

        // 3. Build tree hierarchy of CudaBVHBuildNodes
        CudaBVHBuildNode* dTree = builder->BuildTreeHierarchyHelper(dMortonCodesSorted, dMortonIndicesSorted, nPrimitives);

        // Start measure
        CUDA_SYNC_CHECK();
        state.ResumeTiming();
        
        // 4. Compute Bounding Boxes of each node
	    CudaBVHBuildNode* treeWithBoundingBoxes = builder->ComputeBoundingBoxesHelper(dPrimitiveInfo, dTree, nPrimitives);

        CUDA_SYNC_CHECK();
        state.PauseTiming();
        // End measure

        // Clean Up
        free(treeWithBoundingBoxes);
        delete builder;

        state.ResumeTiming();
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_ComputeBoundingBoxes, 0)->Name("BM_CudaBVHBuilder_ComputeBoundingBoxes/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_ComputeBoundingBoxes, 1)->Name("BM_CudaBVHBuilder_ComputeBoundingBoxes/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_ComputeBoundingBoxes, 2)->Name("BM_CudaBVHBuilder_ComputeBoundingBoxes/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_ComputeBoundingBoxes, 3)->Name("BM_CudaBVHBuilder_ComputeBoundingBoxes/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(true);


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

        const unsigned int nPrimitives = builder->primitiveInfo.size();
	    BVHPrimitiveInfoWithIndex* dPrimitiveInfo = builder->PrepareDevicePrimitiveInfo(nPrimitives);
        
        // 1. Compute Morton Codes
        unsigned int* dMortonCodes;
	    unsigned int* dMortonIndices;
        builder->GenerateMortonCodesHelper(dPrimitiveInfo, &dMortonCodes, &dMortonIndices, nPrimitives);

        // 2. Sort Morton Codes
        unsigned int* dMortonCodesSorted;
	    unsigned int* dMortonIndicesSorted;
	    builder->SortMortonCodesHelper(dPrimitiveInfo, dMortonCodes, dMortonIndices, &dMortonCodesSorted, &dMortonIndicesSorted, nPrimitives);

        // 3. Build tree hierarchy of CudaBVHBuildNodes
        CudaBVHBuildNode* dTree = builder->BuildTreeHierarchyHelper(dMortonCodesSorted, dMortonIndicesSorted, nPrimitives);

        // 4. Compute Bounding Boxes of each node
        CudaBVHBuildNode* treeWithBoundingBoxes = builder->ComputeBoundingBoxesHelper(dPrimitiveInfo, dTree, nPrimitives);

        // Start measure
        CUDA_SYNC_CHECK();
        state.ResumeTiming();
        
        // 5. Flatten Tree and order BVH::primitives according to dMortonIndicesSorted
        builder->PermutePrimitivesAndFlattenTree(dMortonIndicesSorted, treeWithBoundingBoxes, nPrimitives);
       
        CUDA_SYNC_CHECK();
        state.PauseTiming();
        // End measure

        // Clean Up
        delete builder;

        state.ResumeTiming();
    }
    state.counters["primitives"] = scene->numTriangles;
    state.counters["primitives/s"] = benchmark::Counter(scene->numTriangles, benchmark::Counter::kIsRate);
    delete scene;
}

BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree, 0)->Name("BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree/" + SceneNames[0])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree, 1)->Name("BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree/" + SceneNames[1])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree, 2)->Name("BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree/" + SceneNames[2])->Iterations(1)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree, 3)->Name("BM_CudaBVHBuilder_PermutePrimitivesAndFlattenTree/" + SceneNames[3])->Iterations(1)->ReportAggregatesOnly(true);


BENCHMARK_MAIN();

NAMESPACE_DPHPC_END
