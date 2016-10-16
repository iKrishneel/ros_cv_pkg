
#include <kernelized_correlation_filters/discrete_fourier_transform_kernel.h>

__global__
void cuFloatToComplex(cufftComplex *d_complex,
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

    cuFloatToComplex<<<grid_size, block_size>>>(d_complex, dev_data, LENGHT);
    return d_complex;
}
