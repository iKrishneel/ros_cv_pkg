
#include <kernelized_correlation_filters/cosine_convolution_kernel.h>

/*
__host__ __device__
struct caffeFilterInfo {
    int width;
    int height;
    int channels;
    int data_lenght;  //! total lenght = blob->count()
    caffeFilterInfo(int w = -1, int h = -1,
                    int c = -1, int  l = 0) :
       width(w), height(h), channels(c), data_lenght(l) {}
};
*/


__host__ __forceinline__
void cuAssert(cudaError_t code, char *file, int line, bool abort) {
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


__global__ __forceinline__
void cosineConvolutionKernel(float *d_output,
                             const float*d_cnn_codes,
                             const float *d_cos_window,
                             const int data_count) {
   
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < data_count) {
       d_output[offset] = d_cnn_codes[offset] * d_cos_window[offset];

       /*
       if (offset > 109 && offset < 120) {
          printf("%3.4f  %3.4f   %3.4f \n", d_output[offset],
                 d_cnn_codes[offset], d_cos_window[offset]);
       }
       */
    }
}


float* cosineConvolutionGPU(const float *d_cnn_codes,
                            const float *d_cos_window,
                            const int data_count,
                            const int BYTE) {

     const int dimension = std::ceil(std::sqrt(CNN_FILTER_SIZE));
     dim3 block_size(cuDivUp(dimension, GRID_SIZE),
                     cuDivUp(dimension, GRID_SIZE));
     dim3 grid_size(GRID_SIZE, GRID_SIZE);

     float *d_output;
     cudaMalloc(reinterpret_cast<void**>(&d_output), BYTE);
     cosineConvolutionKernel<<<block_size, grid_size>>>(
        d_output, d_cnn_codes, d_cos_window, data_count);

     return d_output;
}


/**
 * bilinar
 */
/*
__device__ __forceinline__
float lerp(float c1, float c2, float v1, float v2, float x) {
    if ((v1 == v2)) {
       return c1;
    }
    float inc = ((c2-c1)/(v2 - v1)) * (x - v1);
    float val = c1 + inc;
    return val;
};


__global__
void bilinearInterpolationKernel(float * d_result,
                                 const float *d_data,
                                 const int nx, const int ny,
                                 const int num_filters,  //! 256
                                 const caffeFilterInfo blob_info) {

    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;
    offset *= num_filters;  // ????
    if (offset < blob_info.data_lenght) {
       
       const float fx = static_cast<float>(blob_info.width)/
          static_cast<float>(nx);
       const float fy = static_cast<float>(blob_info.height)/
          static_cast<float>(ny);
       
       //! indvidual for loops
       for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
             float src_x = i * fx;
             float src_y = j * fy;
             
             int x1 = static_cast<int>(floorf(src_x));
             int y1 = static_cast<int>(floorf(src_y));


             float p1 = d_data[x1 + (y1 * blob_info.width)];
             float p2 = d_data[x1 + 1 + (y1 * blob_info.width)];
             float p3 = d_data[x1 + ((y1 + 1) * blob_info.width)];
             float p4 = d_data[x1 + 1+ ((y1 + 1)* blob_info.width)];


             
             int x2 = x1 + 1;
             int y2 = y1 + 1;

             
             if (i == 10 && j == 10) {
                printf("%d ", x1 + (y1 * blob_info.width));
                printf("%d ", x1 + 1 + (y1 * blob_info.width));
                printf("%d ", x1 + ((y1 + 1) * blob_info.width));
                printf("%d \n", x1 + 1 + ((y1 + 1) * blob_info.width));
                printf("%d \n", blob_info.width);
             }
             
             // const float *d = d_data + x1 + y1 * blob_info.width;
             // float p1 = d[0 + 0 * blob_info.width];
             // float p2 = d[1 + 0 * blob_info.width];
             // float p3 = d[0 + 1 * blob_info.width];
             // float p4 = d[1 + 1 * blob_info.width];

             if (i == 10 && j == 10)
                printf("%3.2f %3.2f %3.2f %3.2f\n", p1, p2, p3, p4);
             
             float wx = i - x1;
             float wy = j - y1;
             float wx1 = 1.0f - wx;
             float wy1 = 1.0f - wy;

             int w1 = wx1 * wy1 * 255.0f;
             int w2 = wx * wy1 * 255.0f;
             int w3 = wx1 * wy * 255.0f;
             int w4 = wx * wy * 255.0f;

             // float out_value = p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4;
             float out_value = (p1 + p2 + p3 + p4)/ 4;

             
             // int x2_read = fminf(x2, blob_info.width - 1);
             // int y2_read = fminf(y2, blob_info.height - 1);
             
             // float src_reg = d_data[(offset + x1 + (y1 * blob_info.width))];
             // float out_1 = src_reg * ((x2 - src_x) * (y1 - src_y));
             // src_reg = d_data[(offset + x2_read + (y1 * blob_info.width))];
             // float out_2 = src_reg * ((src_x - x1) * (y2 - src_y));
             // src_reg = d_data[(offset + x1 + (y2_read * blob_info.width))];
             // float out_3 = src_reg * ((x2 - src_x) * (src_y - y1));
             // src_reg = d_data[(offset + x2_read + (y2_read * blob_info.width))];
             // float out_4 = src_reg * ((src_x - x1) * (src_y - y1));

             
             d_result[offset + i + (j * nx)] = out_value;

          }
       }
    }
}


float *bilinearInterpolationGPU(const float *d_data,
                                const int new_x, const int new_y,
                                const int fwidth, const int fheight,
                                const int flenght,
                                const int num_filters) {
    caffeFilterInfo cfinfo(fwidth, fheight, 1, flenght);
    
    const int dimension = std::ceil(std::sqrt(num_filters));
    dim3 block_size(cuDivUp(dimension, GRID_SIZE),
                    cuDivUp(dimension, GRID_SIZE));
    dim3 grid_size(GRID_SIZE, GRID_SIZE);

    int OUT_BYTE = sizeof(float) * new_y * new_x * num_filters;
    float *d_output;
    cudaMalloc(reinterpret_cast<void**>(&d_output), OUT_BYTE);

    bilinearInterpolationKernel<<<block_size, grid_size>>>(
       d_output, d_data, new_x, new_y, num_filters, cfinfo);

    float *cpu_out = (float*)malloc(OUT_BYTE);
    cudaMemcpy(cpu_out, d_output, OUT_BYTE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < new_y; i++) {
       for (int j = 0; j < new_x; j++) {
          printf("%3.5f ", cpu_out[j + i * new_x]);
       }
       printf("\n");

    }

    printf("SIZE: %d  %d\n", new_x, new_y);
    
    return d_output;
}


float *bilinear_test(float *data, const int in_byte) {

    float *d_data;
    cudaMalloc(reinterpret_cast<void**>(&d_data), in_byte);
    cudaMemcpy(d_data, data, in_byte, cudaMemcpyHostToDevice);

    int new_x = 640;
    int new_y = 480;
    caffeFilterInfo cfinfo(320, 240, 1, 320 * 240);

    int OUT_BYTE = sizeof(float) * new_y * new_x * 1;
    float *d_output;
    cudaMalloc(reinterpret_cast<void**>(&d_output), OUT_BYTE);
    
    bilinearInterpolationKernel<<<1, 1>>>(
       d_output, d_data, new_x, new_y, 1, cfinfo);

    float *cpu_out = (float*)malloc(OUT_BYTE);
    cudaMemcpy(cpu_out, d_output, OUT_BYTE, cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    cudaFree(d_data);
    
    return cpu_out;
}

*/
