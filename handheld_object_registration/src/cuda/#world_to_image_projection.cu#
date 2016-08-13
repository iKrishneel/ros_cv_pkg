
#include <handheld_object_registration/world_to_image_projection.h>

#define CUDA_ERROR_CHECK(process) {                  \
      cudaAssert((process), __FILE__, __LINE__);     \
   }   

void cudaAssert(cudaError_t code, char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
       fprintf(stderr, "GPUassert: %s %s %dn",
               cudaGetErrorString(code), file, line);
       if (abort) {
          exit(code);
       }
    }
}

__host__ __device__ __align__(16)
    int cuDivUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}



