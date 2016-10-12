
#include <kernelized_correlation_filters/cosine_convolution_kernel.h>


__host__ __device__ __align__(16)
   int cuDivUp(
      int a, int b) {

   return ((a % b) != 0) ? (a / b + 1) : (a / b);
}
