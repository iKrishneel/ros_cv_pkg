
#include <kernelized_correlation_filters/bilinear_interpolation_kernel.h>

__global__ __forceinline__
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
       float src_x = 0.0f;
       float src_y = 0.0f;
       float p1 = 0.0f;
       float p2 = 0.0f;
       float p3 = 0.0f;
       float p4 = 0.0f;
       int x1;
       int y1;
       int index_cols = -1;
       int index_rows = -1;
       
       for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
             src_x = i * fx;
             src_y = j * fy;
             x1 = static_cast<int>(floorf(src_x));
             y1 = static_cast<int>(floorf(src_y));

             index_cols = x1 + 0;
             index_rows = y1 + 0;
             p1 = (index_cols > blob_info.width - 1 &&
                   index_rows > blob_info.height - 1) ? 0.0f :
                d_data[index_cols + (index_rows * blob_info.width)];
             index_cols = x1 + 1;
             index_rows = y1 + 0;
             p2 = (index_cols > blob_info.width - 1 &&
                   index_rows > blob_info.height - 1) ? 0.0f :
                d_data[index_cols + (index_rows * blob_info.width)];
             index_cols = x1 + 0;
             index_rows = y1 + 1;
             p3 = (index_cols > blob_info.width - 1 &&
                   index_rows > blob_info.height - 1) ? 0.0f :
                d_data[index_cols + (index_rows * blob_info.width)];
             index_cols = x1 + 1;
             index_rows = y1 + 1;
             p4 = (index_cols > blob_info.width - 1 &&
                   index_rows > blob_info.height - 1) ? 0.0f :
                d_data[index_cols + (index_rows * blob_info.width)];
             
             // p1 = d_data[x1 + 0 + ((y1 + 0) * blob_info.width)];
             // p2 = d_data[x1 + 1 + ((y1 + 0) * blob_info.width)];
             // p3 = d_data[x1 + 0 + ((y1 + 1) * blob_info.width)];
             // p4 = d_data[x1 + 1 + ((y1 + 1) * blob_info.width)];

             float out_value = (p1 + p2 + p3 + p4)/ 4;
             d_result[offset + i + (j * nx)] = out_value;
          }
       }
    }
}

/*--------------TEXTURE--------------------------*/

__global__ __forceinline__
void bilinearInterpolationKernelTexture(float * d_result,
                                        cudaTextureObject_t tex_obj,
                                        const int nx, const int ny,
                                        const int num_filters,  //! 256
                                        const caffeFilterInfo blob_info) {

    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;
    // offset *= num_filters;  // ????
    // if (offset < blob_info.data_lenght) {
    if (offset < num_filters) {

       // offset = offset * blob_info.width * blob_info.height;

       // if (offset < 169 * 3) {
       //    printf("OFFSET: %d\n", offset);
       // }
       
       const float fx = static_cast<float>(blob_info.width)/
          static_cast<float>(nx);
       const float fy = static_cast<float>(blob_info.height)/
          static_cast<float>(ny);
       
       float src_x = 0.0f;
       float src_y = 0.0f;
       float p1 = 0.0f;
       float p2 = 0.0f;
       float p3 = 0.0f;
       float p4 = 0.0f;
       int x1;
       int y1;
       int index_cols = -1;
       int index_rows = -1;

       int arr_index = offset * nx * ny;
       
       for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
             src_x = i * fx;
             src_y = j * fy;
             x1 = static_cast<int>(floorf(src_x));
             y1 = static_cast<int>(floorf(src_y));

             index_cols = x1 + 0;
             index_rows = y1 + 0;
             p1 = (index_cols > blob_info.width - 1 &&
                   index_rows > blob_info.height - 1) ? 0.0f :
                tex1Dfetch<float>(tex_obj,
                                  index_cols + (index_rows * blob_info.width));
             index_cols = x1 + 1;
             index_rows = y1 + 0;
             p2 = (index_cols > blob_info.width - 1 &&
                   index_rows > blob_info.height - 1) ? 0.0f :
                tex1Dfetch<float>(tex_obj,
                                  index_cols + (index_rows * blob_info.width));
             index_cols = x1 + 0;
             index_rows = y1 + 1;
             p3 = (index_cols > blob_info.width - 1 &&
                index_rows > blob_info.height - 1) ? 0.0f :
                tex1Dfetch<float>(tex_obj,
                                  index_cols + (index_rows * blob_info.width));
             index_cols = x1 + 1;
             index_rows = y1 + 1;
             p4 = (index_cols > blob_info.width - 1 &&
                   index_rows > blob_info.height - 1) ? 0.0f :
                tex1Dfetch<float>(tex_obj,
                                  index_cols + (index_rows * blob_info.width));
             
             float out_value = (p1 + p2 + p3 + p4) / 4.0f;
             //! d_result[offset + i + (j * nx)] = out_value;
             //! d_result[offset * nx * ny + i + (j * nx)] = out_value;
             d_result[arr_index + i + (j * nx)] = out_value;
          }
       }
    }
}


float *bilinearInterpolationGPU(const float *d_data,
                                const int new_x, const int new_y,
                                const int filter_width,
                                const int filter_height,
                                const int flenght,  //! data count
                                const int num_filters) {
    caffeFilterInfo cfinfo(filter_width, filter_height, 1, flenght);
    
    const int dimension = std::ceil(std::sqrt(num_filters));
    dim3 grid_size(cuDivUp(dimension, GRID_SIZE),
                    cuDivUp(dimension, GRID_SIZE));
    dim3 block_size(GRID_SIZE, GRID_SIZE);

    const int OUT_BYTE = sizeof(float) * new_y * new_x * num_filters;
    float *d_output;
    cudaMalloc(reinterpret_cast<void**>(&d_output), OUT_BYTE);

    const int IN_BYTE = filter_width * filter_height * sizeof(float);
    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = const_cast<float*>(d_data);
    res_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
    res_desc.res.linear.desc.x = 32;  // bits per channel
    res_desc.res.linear.sizeInBytes = IN_BYTE;
    
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.readMode = cudaReadModeElementType;
    
    cudaTextureObject_t tex_obj = 0;
    cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);

    /*
    bilinearInterpolationKernelTexture<<<grid_size, block_size>>>(
       d_output, tex_obj, new_x, new_y, num_filters, cfinfo);
    */
    bilinearInterpolationKernelTexture<<<num_filters, 1>>>(
       d_output, tex_obj, new_x, new_y, num_filters, cfinfo);

#ifdef _DEBUG
    /*
    bilinearInterpolationKernel<<<grid_size, block_size>>>(
       d_output, d_data, new_x, new_y, num_filters, cfinfo);
    */
    /*
    float *cpu_out = (float*)malloc(OUT_BYTE);
    cudaMemcpy(cpu_out, d_output, OUT_BYTE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < new_y; i++) {
       for (int j = 0; j < new_x; j++) {
          printf("%3.5f ", cpu_out[j + i * new_x]);
       }
       printf("\n");
    }
    printf("SIZE: %d  %d\n", new_x, new_y);
    */
#endif
    
    cudaDestroyTextureObject(tex_obj);
    return d_output;
}



float *bilinear_test(float *data, const int in_byte) {

    float *d_data;
    cudaMalloc(reinterpret_cast<void**>(&d_data), in_byte);
    cudaMemcpy(d_data, data, in_byte, cudaMemcpyHostToDevice);
    
    int new_x = 50;
    int new_y = 50;
    caffeFilterInfo cfinfo(13, 13, 1, 13 * 13);

    //! texture mem
    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = d_data;
    res_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
    res_desc.res.linear.desc.x = 32;  // bits per channel
    res_desc.res.linear.sizeInBytes = in_byte;
    
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.readMode = cudaReadModeElementType;
    
    cudaTextureObject_t tex_obj = 0;
    cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);

    //! end texture
    
    
    int OUT_BYTE = sizeof(float) * new_y * new_x * 1;
    float *d_output;
    cudaMalloc(reinterpret_cast<void**>(&d_output), OUT_BYTE);

    bilinearInterpolationKernelTexture<<<1, 1>>>(
       d_output, tex_obj, new_x, new_y, 1, cfinfo);

    /*
    bilinearInterpolationKernel<<<1, 1>>>(
       d_output, d_data, new_x, new_y, 1, cfinfo);
    */
    
    
    float *cpu_out = (float*)malloc(OUT_BYTE);
    cudaMemcpy(cpu_out, d_output, OUT_BYTE, cudaMemcpyDeviceToHost);

    cudaDestroyTextureObject(tex_obj);
    cudaFree(d_output);
    cudaFree(d_data);
    
    return cpu_out;
}

