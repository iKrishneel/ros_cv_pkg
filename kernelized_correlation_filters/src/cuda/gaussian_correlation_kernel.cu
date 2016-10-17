
#include <kernelized_correlation_filters/gaussian_correlation_kernel.h>


__device__ __forceinline__
float squaredMagnitude(const cufftComplex data) {
    return (powf(data.x, 2) + powf(data.y, 2));
}

__global__ __forceinline__
void squaredNormKernel(float *d_squared_norm,
                       const cufftComplex *d_complex,
                       const int LENGHT) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < LENGHT) {
       // d_squared_norm[offset] = squaredMagnitude(d_complex[offset]);
       d_squared_norm[offset] = (d_complex[offset].x * d_complex[offset].x) +
          (d_complex[offset].y * d_complex[offset].y);

       /*
       if (isnan(d_squared_norm[offset])) {
          printf("GPU DEBUG: %d  %3.5f  %3.5f\n", offset,
                 d_complex[offset].x, d_complex[offset].y);
       }
       */
    }
}



float squaredNormGPU(const cufftComplex *d_complex,
                     const int FILTER_BATCH,
                     const int FILTER_SIZE) {
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [squaredNormGPU] FAILED\n");
    }
    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    
    float *d_squared_norm;
    const int BYTE = LENGHT * sizeof(float);
    cudaMalloc(reinterpret_cast<void**>(&d_squared_norm), BYTE);

    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                   cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);
    squaredNormKernel<<<grid_size, block_size>>>(d_squared_norm,
                                                 d_complex, LENGHT);
    
    float *d_summation;
    cudaMalloc(reinterpret_cast<void**>(&d_summation), BYTE);

    // TODO(TX1):  check and set auto
    int num_threads = 128;
    int num_blocks = 64;

    reduceSinglePass(LENGHT, num_threads, num_blocks,
                     d_squared_norm, d_summation);

    float *sum = reinterpret_cast<float*>(std::malloc(BYTE));
    cudaMemcpy(sum, d_summation, BYTE, cudaMemcpyDeviceToHost);
    
    float norm = sum[0] / FILTER_SIZE;
    
    free(sum);
    cudaFree(d_squared_norm);
    cudaFree(d_summation);

    return norm;
}


/**
 * kernel for computing just the inverse
 */
__global__ __forceinline__
void invComplexConjuateKernel(cufftComplex *d_compl_out,
                              const cufftComplex *d_complex,
                              const int LENGHT) {

    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < LENGHT) {
       d_compl_out[offset] = d_complex[offset];
       d_compl_out[offset].y *= -1.0f;
    }
   
}

cufftComplex* invComplexConjuateGPU(const cufftComplex *d_complex,
                                    const int FILTER_BATCH,
                                    const int FILTER_SIZE) {

    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [invComplexConjuateGPU] FAILED\n");
    }
    
    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                   cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    const int BYTE = LENGHT * sizeof(cufftComplex);
    cufftComplex *d_compl_out;
    cudaMalloc(reinterpret_cast<void**>(&d_compl_out), BYTE);

    invComplexConjuateKernel<<<grid_size, block_size>>>(d_compl_out,
                                                        d_complex, LENGHT);
    
    return d_compl_out;
}


/**
 * kernel to inverse and multipy reduced into one
 */

__global__ __forceinline__
void invConjuateConvKernel(cufftComplex *d_compl_out,
                           const cufftComplex *d_complex,
                           const cufftComplex *d_compl_model,
                           const int LENGHT) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < LENGHT) {
       d_compl_out[offset].x = (
          (d_complex[offset].x * d_compl_model[offset].x) -
          (d_complex[offset].y * (d_compl_model[offset].y * -1.0f)));
       d_compl_out[offset].y = 0.0f;
    }
}

cufftComplex* invConjuateConvGPU(const cufftComplex *d_complex,
                                 const cufftComplex *d_compl_model,
                                 const int FILTER_BATCH,
                                 const int FILTER_SIZE) {

    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [invConjuateConvGPU] FAILED\n");
    }
    
    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                   cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    const int BYTE = LENGHT * sizeof(cufftComplex);
    cufftComplex *d_compl_out;
    cudaMalloc(reinterpret_cast<void**>(&d_compl_out), BYTE);
    
    invConjuateConvKernel<<<grid_size, block_size>>>(d_compl_out, d_complex,
                                                     d_compl_model, LENGHT);
    
    return d_compl_out;
}

