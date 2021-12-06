#include "benchmark/benchmark.h"

static void BM_TestBenchmarkFramework(benchmark::State& state) {
    volatile int sum = 0;
    for (auto _ : state) {
        for (size_t i = 0; i < 1e5; i++) {
                sum += i;
        } 
    }
}
BENCHMARK(BM_TestBenchmarkFramework)->Iterations(1)->ReportAggregatesOnly(true);

BENCHMARK_MAIN();