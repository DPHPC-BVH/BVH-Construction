#pragma once
#include <cuda_runtime.h>
#include <chrono>


class Timer {
public:
	virtual void Start() = 0;
	// Return duration in micro seconds (us)
	virtual float Stop() = 0;
};


class TimerCPU : public Timer {
public:
	void Start() override;

	float Stop() override;

private:
	std::chrono::high_resolution_clock::time_point start;

};

class TimerGPU : public Timer {
public:
	


	void Start() override;


	float Stop() override;

private:
	cudaEvent_t start;
	cudaEvent_t stop;
};