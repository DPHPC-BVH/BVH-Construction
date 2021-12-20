#!/bin/bash

REPETITIONS=30
WARMUPS=2
RUNS=$((REPETITIONS + WARMUPS))
CLUSTER='false'

OUT_DIR="out/"
BENCH_OUT_DIR=$OUT_DIR"benchmarks/"

mkdir -p $BENCH_OUT_DIR

declare -a bmrks_infos=(
    "conference;BM_CudaBVHBuilderAlgorithmOnly.*conference"
    "fairyforest;BM_CudaBVHBuilderAlgorithmOnly.*fairyforest"
    "sibenik;BM_CudaBVHBuilderAlgorithmOnly.*sibenik"
    "sanmiguel;BM_CudaBVHBuilderAlgorithmOnly.*sanmiguel"
    "conference_stages;BM_CudaBVHBuilder_(GenerateMortonCodes|SortMortonCodes|BuildTreeHierarchy|ComputeBoundingBoxes).*conference"
    "fairyforest_stages;BM_CudaBVHBuilder_(GenerateMortonCodes|SortMortonCodes|BuildTreeHierarchy|ComputeBoundingBoxes).*fairyforest"
    "sibenik_stages;BM_CudaBVHBuilder_(GenerateMortonCodes|SortMortonCodes|BuildTreeHierarchy|ComputeBoundingBoxes).*sibenik"
    "sanmiguel_stages;BM_CudaBVHBuilder_(GenerateMortonCodes|SortMortonCodes|BuildTreeHierarchy|ComputeBoundingBoxes).*sanmiguel"
    "conference_bounding_boxes;BM_CudaBVHBuilder_ComputeBoundingBoxes/.*conference"
    "fairyforest_bounding_boxes;BM_CudaBVHBuilder_ComputeBoundingBoxes/.*fairyforest"
    "sibenik_bounding_boxes;BM_CudaBVHBuilder_ComputeBoundingBoxes/.*sibenik"
    "sanmiguel_bounding_boxes;BM_CudaBVHBuilder_ComputeBoundingBoxes/.*sanmiguel"
    "conference_bounding_boxes_shared_memory;BM_CudaBVHBuilder_ComputeBoundingBoxesWithSharedMemory.*conference"
    "fairyforest_bounding_boxes_shared_memory;BM_CudaBVHBuilder_ComputeBoundingBoxesWithSharedMemory.*fairyforest"
    "sibenik_bounding_boxes_shared_memory;BM_CudaBVHBuilder_ComputeBoundingBoxesWithSharedMemory.*sibenik"
    "sanmiguel_bounding_boxes_shared_memory;BM_CudaBVHBuilder_ComputeBoundingBoxesWithSharedMemory.*sanmiguel"
)

for b in ${bmrks_infos[@]};
do  
    IFS=";" read -r -a arr <<< "${b}"
    NAME=${arr[0]}
    REGEX=${arr[1]}

    if [[ "${CLUSTER}" == "true" ]]; then
        bash ./submit.sh "./build/Bench/bench_dphp_bvh --benchmark_repetitions=${RUNS} --benchmark_counters_tabular=true --benchmark_out_format=csv --benchmark_out=${BENCH_OUT_DIR}${NAME}.csv --benchmark_filter=\"${REGEX}\""
    else
        ./build/Bench/bench_dphp_bvh --benchmark_repetitions=${RUNS} --benchmark_counters_tabular=true --benchmark_out_format=csv --benchmark_out=${BENCH_OUT_DIR}${NAME}.csv --benchmark_filter="${REGEX}"
    fi
done

