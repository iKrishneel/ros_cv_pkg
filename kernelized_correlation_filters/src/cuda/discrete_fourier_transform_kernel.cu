
#include <kernelized_correlation_filters/discrete_fourier_transform_kernel.h>

__global__
void cuFloatToComplexKernel(cufftComplex *d_complex,
                      const float *dev_data, const int lenght) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < lenght) {
       d_complex[offset].x = dev_data[offset];
       d_complex[offset].y = 0.0f;
    }
}

cufftComplex* convertFloatToComplexGPU(const float *dev_data,
                                       const int FILTER_BATCH,
                                       const int FILTER_SIZE) {
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [convertFloatToComplexGPU] FAILED\n");
    }
    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    const int BYTE = LENGHT * sizeof(cufftComplex);
    cufftComplex *d_complex;
    cudaMalloc(reinterpret_cast<void**>(&d_complex), BYTE);

    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                    cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    cuFloatToComplexKernel<<<grid_size, block_size>>>(
       d_complex, dev_data, LENGHT);
    return d_complex;
}

__global__
void copyComplexRealToFloatKernel(float *d_output,
                                  const cufftComplex *d_complex,
                                  const int lenght) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < lenght) {
       d_output[offset] = d_complex[offset].x;
    }
}

float* copyComplexRealToFloatGPU(const cufftComplex* d_complex,
                                const int FILTER_BATCH,
                                const int FILTER_SIZE) {
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [copyComplexRealToFloatGPU] FAILED\n");
    }
    int LENGHT = FILTER_SIZE * FILTER_BATCH;
    int BYTE = LENGHT * sizeof(float);

    float *d_output;
    cudaMalloc(reinterpret_cast<void**>(&d_output), BYTE);

    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                    cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);
    
    copyComplexRealToFloatKernel<<<grid_size, block_size>>>(
       d_output, d_complex, LENGHT);

    return d_output;
}


__global__
void normalizeByFactorKernel(float *d_data,
                             const float factor,
                             const int lenght) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < lenght) {
       d_data[offset] /= factor;
    }
}

void normalizeByFactorGPU(float *&d_data,
                          const float factor,
                          const int FILTER_BATCH,
                          const int FILTER_SIZE) {
    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       printf("\033[31m ERROR: [normalizeByFactorGPU] FAILED\n");
    }

    int LENGHT = FILTER_BATCH * FILTER_SIZE;
    const int BYTE = LENGHT * sizeof(float);
    const int dimension = std::ceil(std::sqrt(LENGHT));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                    cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    normalizeByFactorKernel<<<grid_size, block_size>>>(
       d_data, factor, LENGHT);
}
