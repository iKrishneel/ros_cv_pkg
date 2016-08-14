
#include <handheld_object_registration/projected_correspondences.h>


__device__ struct cuRect{
    int x;
    int y;
    int width;
    int height;

    // cuRect(int i = 0, int j = 0, int w = 0, int h = 0) :
    //    x(i), y(j), width(w), height(h) {}
};


#define CUDA_ERROR_CHECK(process) {                  \
      cudaAssert((process), __FILE__, __LINE__);     \
   }                                                 \

void cudaAssert(cudaError_t code, char *file, int line, bool abort) {
    if (code != cudaSuccess) {
       fprintf(stderr, "GPUassert: %s %s %dn",
               cudaGetErrorString(code), file, line);
       if (abort) {
          exit(code);
       }
    }
}

__host__ __device__ __align__(16)
    int cuDivUp(
       int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__ __forceinline__
void cuConditionROI(cuRect *rect, int width, int height) {
    if (rect->x < 0) {
       rect->x = 0;
    }
    if (rect->y < 0) {
      rect->y = 0;
    }
    if ((rect->width + rect->x) > width) {
       rect->x -= ((rect->width + rect->x) - width);
    }
    if ((rect->height + rect->y) > height) {
       rect->y -= ((rect->height + rect->y) - height);
    }
}

__device__ __forceinline__
float cuEuclideanDistance(float *src_pt, float *model_pt, const int lenght) {
    float sum = 0.0f;
    for (int i = 0; i < lenght; i++) {
       sum += ((src_pt[i] - model_pt[i]) * (src_pt[i] - model_pt[i]));
    }
    if (sum == 0.0f) {
       return sum;
    }
    return sqrtf(sum);
}

__global__ __forceinline__
void findCorrespondencesGPU(Correspondence * correspondences,
                            cuMat<float, NUMBER_OF_ELEMENTS> *d_src_points,
                            int *d_src_indices,
                            cuMat<float, NUMBER_OF_ELEMENTS> *d_model_points,
                            // cuMat<int, 2> *d_model_indices,
                            int *d_model_indices,
                            const int im_width, const int im_height,
                            const int model_size, const int wsize) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;
    
    if (offset < im_width * im_height) {
       int model_index = d_model_indices[offset];
       if (model_index != -1) {
          cuRect rect;
          rect.x = t_idx - wsize/2;
          rect.y = t_idy - wsize/2;
          rect.width = wsize;
          rect.height = wsize;
          cuConditionROI(&rect, im_width, im_height);

          float model_pt[3];
          model_pt[0] = d_model_points[model_index].data[0];
          model_pt[1] = d_model_points[model_index].data[1];
          model_pt[2] = d_model_points[model_index].data[2];
          
          float min_dsm = FLT_MAX;
          int min_ism = -1;
          for (int l = rect.y; l < rect.y + rect.height; l++) {
             for (int k = rect.x; k < rect.x + rect.width; k++) {
                int src_index = d_src_indices[k + (l * im_width)];
                if (src_index != -1) {
                   float src_pt[3];
                   src_pt[0] = d_src_points[src_index].data[0];
                   src_pt[1] = d_src_points[src_index].data[1];
                   src_pt[2] = d_src_points[src_index].data[2];
                   float dist = cuEuclideanDistance(src_pt, model_pt, 3);
                   
                   if (dist < min_dsm) {
                      min_dsm = dist;
                      min_ism = src_index;
                   }
                }
             }
          }
          if (min_ism != -1 && min_dsm < DISTANCE_THRESH) {
             correspondences[model_index].query_index = model_index;
             correspondences[model_index].match_index = min_ism;
          } else {
             correspondences[model_index].query_index = -1;
             correspondences[model_index].match_index = -1;
          }
       }
    }
}

void estimatedCorrespondences(
    const pcl::PointCloud<PointTYPE>::Ptr source_points,
    const ProjectionMap &src_projection,
    const pcl::PointCloud<PointTYPE>::Ptr target_points,
    const ProjectionMap &target_projection) {
    if (source_points->empty() || target_points->empty()) {
       printf("\033[31m EMPTY POINTCLOUD FOR CORRESPONDENCES \033[0m\n");
       return;
    }
    
    // const int TGT_SIZE = target_projection.indices.rows *
    //    target_projection.indices.cols;
    const int TGT_SIZE = target_projection.width * target_projection.height;
    cuMat<float, NUMBER_OF_ELEMENTS> model_points[TGT_SIZE];
    
    const int SRC_SIZE = src_projection.indices.rows *
       src_projection.indices.cols;
    cuMat<float, NUMBER_OF_ELEMENTS> src_points[SRC_SIZE];
    int src_indices[SRC_SIZE];
    int model_indices[SRC_SIZE];
    
    for (int j = 0; j < target_projection.indices.rows; j++) {
       for (int i = 0; i < target_projection.indices.cols; i++) {
          int index = target_projection.indices.at<int>(j, i);
          if (index != -1) {
             model_points[index].data[0] = target_points->points[index].x;
             model_points[index].data[1] = target_points->points[index].y;
             model_points[index].data[2] = target_points->points[index].z;
          }
          int idx = i + (j * src_projection.indices.cols);
          model_indices[idx] = index;
          
          index = -1;
          index = src_projection.indices.at<int>(j, i);
          
          if (index != -1) {
             src_points[index].data[0] = source_points->points[index].x;
             src_points[index].data[1] = source_points->points[index].y;
             src_points[index].data[2] = source_points->points[index].z;
          }
          src_indices[idx] = index;
       }
    }
    
    dim3 block_size(cuDivUp(target_projection.indices.cols, GRID_SIZE),
                    cuDivUp(target_projection.indices.rows, GRID_SIZE));
    dim3 grid_size(GRID_SIZE, GRID_SIZE);

    size_t TMP_SIZE = TGT_SIZE * sizeof(cuMat<float, 3>);
    cuMat<float, NUMBER_OF_ELEMENTS> *d_model_points;
    cudaMalloc(reinterpret_cast<void**>(&d_model_points), TMP_SIZE);
    cudaMemcpy(d_model_points, model_points, TMP_SIZE, cudaMemcpyHostToDevice);

    size_t TIP_SIZE = SRC_SIZE * sizeof(int);
    int *d_model_indices;
    cudaMalloc(reinterpret_cast<void**>(&d_model_indices), TIP_SIZE);
    cudaMemcpy(d_model_indices, model_indices, TIP_SIZE,
               cudaMemcpyHostToDevice);

    // size_t TIP_SIZE = TGT_SIZE * sizeof(cuMat<float, 2>);
    // cuMat<int, 2> *d_model_indices;
    // cudaMalloc(reinterpret_cast<void**>(&d_model_indices), TIP_SIZE);
    // cudaMemcpy(d_model_indices, model_indices, TIP_SIZE,
    //            cudaMemcpyHostToDevice);
    
    size_t SMP_SIZE = SRC_SIZE * sizeof(cuMat<float, 3>);
    cuMat<float, NUMBER_OF_ELEMENTS> *d_src_points;
    cudaMalloc(reinterpret_cast<void**>(&d_src_points), SMP_SIZE);
    cudaMemcpy(d_src_points, src_points, SMP_SIZE, cudaMemcpyHostToDevice);

    // size_t SIP_SIZE = SRC_SIZE * sizeof(cuMat<float, 1>);
    size_t SIP_SIZE = SRC_SIZE * sizeof(int);
    // cuMat<int, 1> *d_src_indices;
    int *d_src_indices;
    cudaMalloc(reinterpret_cast<void**>(&d_src_indices), SIP_SIZE);
    cudaMemcpy(d_src_indices, src_indices, SIP_SIZE,
               cudaMemcpyHostToDevice);

    Correspondence *d_correspondences;
    cudaMalloc(reinterpret_cast<void**>(&d_correspondences),
               sizeof(Correspondence) * TGT_SIZE);
    
    findCorrespondencesGPU<<<block_size, grid_size>>>(
       d_correspondences, d_src_points, d_src_indices, d_model_points,
       d_model_indices, src_projection.indices.cols,
       src_projection.indices.rows,
       target_projection.width * target_projection.height, 16);


    Correspondence correspondences[TGT_SIZE];
    cudaMemcpy(correspondences, d_correspondences,
               sizeof(Correspondence) * TGT_SIZE, cudaMemcpyDeviceToHost);
    
    // for (int i = 0; i < TGT_SIZE; i++) {
    //    if (correspondences[i].query_index != -1 &&
    //        correspondences[i].match_index != -1) {
    //       std::cout << correspondences[i].query_index << "\t"
    //                 << correspondences[i].match_index << "\n";
    //    }
    // }
    
    cudaFree(d_src_indices);
    cudaFree(d_src_points);
    cudaFree(d_correspondences);
    cudaFree(d_model_points);
    cudaFree(d_model_indices);

    // std::cout << "EXITING..."  << "\n";
    // exit(-1);
}
