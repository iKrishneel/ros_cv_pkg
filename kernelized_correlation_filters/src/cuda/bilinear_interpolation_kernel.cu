
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
       int index = -1;
       float src_y = 0.0f;
       float src_x = 0.0f;
       for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
             src_x = i * fx;
             src_y = j * fy;
             
             int x1 = __float2int_rd(src_x);
             int y1 = __float2int_rd(src_y);

             int x2 = x1 + 1;
             int y2 = y1 + 1;

             int x2_read = fminf(x2, blob_info.width - 1);
             int y2_read = fminf(y2, blob_info.height - 1);

             float out_value = 0.0f;
             float src_reg = d_data[(offset + x1 + (y1 * blob_info.width))];
             out_value += src_reg * ((x2 - src_x) * (y1 - src_y));
             src_reg = d_data[(offset + x2_read + (y1 * blob_info.width))];
             out_value += src_reg * ((src_x - x1) * (y2 - src_y));
             src_reg = d_data[(offset + x1 + (y2_read * blob_info.width))];
             out_value += src_reg * ((x2 - src_x) * (src_y - y1));
             src_reg = d_data[(offset + x2_read + (y2_read * blob_info.width))];
             out_value += src_reg * ((src_x - x1) * (src_y - y1));
             
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
