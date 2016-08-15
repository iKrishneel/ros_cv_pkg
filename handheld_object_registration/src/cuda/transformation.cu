
#include <handheld_object_registration/transformation.h>

#define ELEMENTS 12

template<class T, int N> struct __align__(16) cuMat{
    T data[N];
};

__host__ __device__ __align__(16)
    int icuDivUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ __forceinline__
void transformationKernel(float *d_points_out,
                          float *d_points,
                          float* d_transform,
                          const int lenght) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;
    offset *= ELEMENTS;
    
    if (offset < lenght) {
       d_points_out[offset + 3] = d_points[offset + 3];
       for (int i = 7; i < ELEMENTS; i++) {
          d_points_out[offset + i] = d_points[offset + i];
       }
       /* points */
       d_points_out[offset + 0] =
          d_transform[0] * d_points[offset + 0] +
          d_transform[1] * d_points[offset + 1] +
          d_transform[2] * d_points[offset + 2] +
          d_transform[3];
       
       d_points_out[offset + 1] =
          d_transform[4] * d_points[offset + 0] +
          d_transform[5] * d_points[offset + 1]  +
          d_transform[6] * d_points[offset + 2] +
          d_transform[7];
       
       d_points_out[offset + 2] =
          d_transform[8] * d_points[offset + 0] +
          d_transform[9] * d_points[offset + 1] +
          d_transform[10] * d_points[offset + 2] +
          d_transform[11];

       d_points_out[offset + 4] =
          d_transform[0] * d_points[offset + 4] +
          d_transform[1] * d_points[offset + 5] +
          d_transform[2] * d_points[offset + 6];
       
       d_points_out[offset + 5] =
          d_transform[4] * d_points[offset + 4] +
          d_transform[5] * d_points[offset + 5] +
          d_transform[6] * d_points[offset + 6];
       
       d_points_out[offset + 6] =
          d_transform[8] * d_points[offset + 4] +
          d_transform[9] * d_points[offset + 5] +
          d_transform[10] * d_points[offset + 6];
    }

    // if (offset  == 0) {
    //    printf("GPU : %3.4f, %3.4f, %3.4f\n",
    //           d_points_out[offset + 0], d_points_out[offset + 1],
    //       d_points_out[offset + 2]);
    // }
}

void transformPointCloudWithNormalsGPU(
    pcl::PointCloud<PointTYPE>::Ptr in_cloud,
    pcl::PointCloud<PointTYPE>::Ptr out_cloud,
    const Eigen::Matrix4f in_transform) {
    if (in_cloud->empty()) {
       printf("\033[31m EMPTY POINTCLOUD FOR TRANSFORMATION \033[0m\n");
       return;
    }

    std::cout << "\033[35m IN POINT SIZE: \033[0m"  << in_cloud->size() << "\n";
    
    float *transform_vector = reinterpret_cast<float*>(
       std::malloc(sizeof(float) * 16));
    for (int i = 0; i < 4; i++) {
       for (int j = 0; j < 4; j++) {
          transform_vector[j + (i * 4)] = in_transform(i, j);
       }
    }
    float *d_transform;
    cudaMalloc(reinterpret_cast<void**>(&d_transform), 16 * sizeof(float));
    cudaMemcpy(d_transform, transform_vector, 16 * sizeof(float),
               cudaMemcpyHostToDevice);
    
    const int IN_SIZE = in_cloud->size() * ELEMENTS;
    // float *data = reinterpret_cast<float*>(
    //    std::malloc(sizeof(float) * IN_SIZE));
    // std::memcpy(in_cloud->points.data(), data, sizeof(float) * IN_SIZE);

    float *d_data;
    cudaMalloc(reinterpret_cast<void**>(&d_data),
               IN_SIZE * sizeof(float));
    cudaMemcpy(d_data, in_cloud->points.data(),
               IN_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);
    /* packing order x, y, z, 0, nx, ny, nz*/

    float *d_points_out;
    cudaMalloc(reinterpret_cast<void**>(&d_points_out),
               IN_SIZE * sizeof(float));
    
    dim3 block_size(icuDivUp(128, GRID_SIZE),
                    icuDivUp(96, GRID_SIZE));
    dim3 grid_size(GRID_SIZE, GRID_SIZE);
    
    transformationKernel<<<block_size, grid_size>>>(
       d_points_out, d_data, d_transform, IN_SIZE);
    
    out_cloud->height = in_cloud->height;
    out_cloud->width = in_cloud->width;
    out_cloud->header = in_cloud->header;
    out_cloud->resize(in_cloud->size());
    
    cudaMemcpy(out_cloud->points.data(), d_points_out,
               IN_SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);

    // cuMat<float, ELEMENTS> *points = reinterpret_cast<cuMat<float, ELEMENTS>*>(
    //    std::malloc(sizeof(cuMat<float, ELEMENTS>) * IN_SIZE));
    // for (int i = 0; i < IN_SIZE; i++) {
    //    points[i].data[0] = in_cloud->points[i].x;
    //    points[i].data[1] = in_cloud->points[i].y;
    //    points[i].data[2] = in_cloud->points[i].z;
    //    points[i].data[3] = in_cloud->points[i].normal_x;
    //    points[i].data[4] = in_cloud->points[i].normal_y;
    //    points[i].data[5] = in_cloud->points[i].normal_z;
    // }

    
    free(transform_vector);
    cudaFree(d_transform);
    cudaFree(d_data);
    cudaFree(d_points_out);
    // free(data);
}
