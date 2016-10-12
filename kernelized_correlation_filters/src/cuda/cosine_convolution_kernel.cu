
#include <kernelized_correlation_filters/cosine_convolution_kernel.h>

__host__ __device__
struct caffeFilterInfo {
    int width;
    int height;
    int channels;
    int data_lenght;
    caffeFilterInfo(int w = -1, int h = -1,
                    int c = -1, int  l = 0) :
       width(w), height(h), channels(c), data_lenght(l) {}
};

__global__ __forceinline__
void cosineConvolutionKernel(float *d_output,
                             const float*d_cnn_codes,
                             const float *d_cos_window,
                             const caffeFilterInfo info) {
   
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < info.data_lenght) {
       
       for (int j = 0; j < info.height; j++) {
          for (int i = 0; i < info.width; i++) {
             
          }
       }
    }
}

__host__ __forceinline__
void cosineConvolutionGPU(float *d_output,
                          const float *d_cnn_codes,
                          const cv::cuda::GpuMat cos_window,
                          const int filter_lenght) {

    float *d_cos_window;
    const int BYTE = cos_window.rows * cos_window.step;
    cudaMalloc(reinterpret_cast<void**>(&d_cos_window), BYTE);
    cudaMemcpy(d_cos_window, cos_window.data, BYTE, cudaMemcpyDeviceToDevice);
    
    const int dimension = std::ceil(std::sqrt(CNN_FILTER_SIZE));
    dim3 block_size(cuDivUp(dimension, GRID_SIZE),
                    cuDivUp(dimension, GRID_SIZE));
    dim3 grid_size(GRID_SIZE, GRID_SIZE);
    
    
}

