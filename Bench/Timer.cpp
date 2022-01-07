#include "Timer.h"
#include <chrono>
#include <cuda_runtime.h>

void TimerCPU::Start() {
	start = std::chrono::high_resolution_clock::now();
}

float TimerCPU::Stop() {
	auto duration = std::chrono::high_resolution_clock::now() - start;
	return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() / 1000.0f;
}

void TimerGPU::Start() {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

float TimerGPU::Stop() {
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	return ms * 1000.0f;
}