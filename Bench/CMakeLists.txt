set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)

add_subdirectory(benchmark)
include_directories(benchmark/include)

set(bench_cc
  TestBenchmarkFramework.cc
)

add_executable(bench_dphp_bvh ${bench_cc})

target_link_libraries(bench_dphp_bvh
  benchmark
  ${BVH_LIB_NAME}
  ${CMAKE_THREAD_LIBS_INIT}
)