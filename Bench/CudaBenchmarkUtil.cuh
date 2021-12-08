#pragma once

#include "DPHPC.h"
#include <cuda.h>
#include <cuda_runtime.h>

NAMESPACE_DPHPC_BEGIN

/**
 * This macro make sure that all issued tasks for the devive are completed.
 */
#define CUDA_SYNC_CHECK()                                               \
  {                                                                     \
    cudaDeviceSynchronize();                                            \
    cudaError_t error = cudaGetLastError();                             \
    if( error != cudaSuccess )                                          \
      {                                                                 \
        fprintf( stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString( error ) ); \
        exit( 2 );                                                      \
      }                                                                 \
  }

/**
 * Warms up GPU for benchmarking
 */
void WarmUpGPU();
__global__ void WarmUpGPUKernel();



NAMESPACE_DPHPC_END
