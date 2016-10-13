
#ifndef _CUDA_COMMON_H_
#define _CUDA_COMMON_H_

#include <stdio.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cufft.h>

#define GRID_SIZE 16
#define CNN_FILTER_SIZE 256
/*
#define CUDA_ERROR_CHECK(process) {                    \
      cudaAssert((process), __FILE__, __LINE__);       \
   }                                                   \
      
      
void cudaAssert(cudaError_t code, char *file, int line, bool abort) {
    if (code != cudaSuccess) {
       fprintf(stderr, "GPUassert: %s %s %dn",
               cudaGetErrorString(code), file, line);
       if (abort) {
          exit(code);
      }
    }
}

*/
// __host__ __device__ __align__(16)
int cuDivUp(int a, int b);
/*
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}
*/
#endif  // _CUDA_COMMON_H_ 

