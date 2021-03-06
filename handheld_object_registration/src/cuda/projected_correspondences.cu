
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
float cuEuclideanDistance(float *src_pt, float *model_pt,
                          const int lenght, bool sqrt_dist = true) {
    float sum = 0.0f;
    for (int i = 0; i < lenght; i++) {
       sum += ((src_pt[i] - model_pt[i]) * (src_pt[i] - model_pt[i]));
    }
    if (sqrt_dist) {
       return sqrtf(sum);
    } else {
       return sum;
    }
}

__global__ __forceinline__
void findCorrespondencesGPU(Correspondence * correspondences,
                            cuMat<float, NUMBER_OF_ELEMENTS> *d_src_points,
                            int *d_src_indices,
                            cuMat<float, NUMBER_OF_ELEMENTS> *d_model_points,
                            int *d_model_indices, const int im_width,
                            const int im_height, const int wsize) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    //! temp
    /*
    if (offset == 0) {
       for (int i = 0; i < im_width * im_height; i++) {
          correspondences[i].query_index = -1;
          correspondences[i].match_index = -1;
       }
    }
    __syncthreads();
    */
    if (offset < im_width * im_height) {
       int model_index = d_model_indices[offset];
       if (model_index != -1) {
          cuRect rect;
          rect.x = t_idx - wsize/2;
          rect.y = t_idy - wsize/2;
          rect.width = wsize;
          rect.height = wsize;
          cuConditionROI(&rect, im_width, im_height);

#ifdef _DEBUG
          if (model_index == 10) {
             printf("\n\nRECT: %d, %d, %d, %d\n",
                    rect.x, rect.y, rect.width, rect.height);
          }
#endif
          
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
                   
                   if (dist >= 0.0f && dist < min_dsm && !isnan(dist)
                      && dist < DISTANCE_THRESH) {
                      min_dsm = dist;
                      min_ism = src_index;
                   }
                }
             }
          }
          if (min_ism != -1 && min_dsm < DISTANCE_THRESH) {
             correspondences[model_index].query_index = model_index;
             correspondences[model_index].match_index = min_ism;
             correspondences[model_index].distance = min_dsm;
          } else {
             correspondences[model_index].query_index = -1;
             correspondences[model_index].match_index = -1;
          }
       }
    }
}

// cuMat<float, NUMBER_OF_ELEMENTS> *src_points;
// cuMat<float, NUMBER_OF_ELEMENTS> *d_src_points;
// int *d_src_indices;

bool allocateCopyDataToGPU(
    pcl::Correspondences &corr, float &energy,
    bool allocate_src,
    const pcl::PointCloud<PointTYPE>::Ptr source_points,
    const ProjectionMap &src_projection,
    const pcl::PointCloud<PointTYPE>::Ptr target_points,
    const ProjectionMap &target_projection) {
    if (source_points->empty() || target_points->empty()) {
       printf("\033[31m EMPTY POINTCLOUD FOR CORRESPONDENCES \033[0m\n");
       return false;
    }

    int *d_src_indices;
    int *d_model_indices;
    cuMat<float, NUMBER_OF_ELEMENTS> *d_model_points;
    cuMat<float, NUMBER_OF_ELEMENTS> *d_src_points;
    
    const int TGT_SIZE = std::max(
       static_cast<int>(target_points->size()),
       target_projection.width * target_projection.height);
    cuMat<float, NUMBER_OF_ELEMENTS> model_points[TGT_SIZE];

    const int SRC_POINT_SIZE = static_cast<float>(source_points->size());
    // cuMat<float, NUMBER_OF_ELEMENTS> src_points[SRC_POINT_SIZE];
    cuMat<float, NUMBER_OF_ELEMENTS> *src_points;

    if (allocate_src) {
       src_points = reinterpret_cast<cuMat<float, NUMBER_OF_ELEMENTS>* >(
       std::malloc(sizeof(cuMat<float, NUMBER_OF_ELEMENTS>) * SRC_POINT_SIZE));
    }
    
    const int SRC_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
    int src_indices[SRC_SIZE];
    int model_indices[SRC_SIZE];

    int image_size = IMAGE_WIDTH * IMAGE_HEIGHT;
    Correspondence *correspondences = reinterpret_cast<Correspondence*>(
       std::malloc(sizeof(Correspondence) * image_size));
    
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(8) collapse(2)
// #endif
    for (int j = 0; j < target_projection.indices.rows; j++) {
       for (int i = 0; i < target_projection.indices.cols; i++) {
          int idx = i + (j * src_projection.indices.cols);

          correspondences[idx].query_index = -1;
          correspondences[idx].match_index = -1;
          correspondences[idx].distance = FLT_MAX;
          
          int index = target_projection.indices.at<int>(j, i);
          if (index != -1) {
             float x = target_points->points[index].x;
             float y = target_points->points[index].y;
             float z = target_points->points[index].z;
             if (!isnan(x) && !isnan(y) && !isnan(z)) {
                model_points[index].data[0] = x;
                model_points[index].data[1] = y;
                model_points[index].data[2] = z;
             } else {
                index = -1;
             }
          }
          model_indices[idx] = index;

          if (allocate_src) {
             index = -1;
             index = src_projection.indices.at<int>(j, i);
          
             if (index != -1) {
                float x = source_points->points[index].x;
                float y = source_points->points[index].y;
                float z = source_points->points[index].z;

                if (!isnan(x) && !isnan(y) && !isnan(z)) {
                   src_points[index].data[0] = x;
                   src_points[index].data[1] = y;
                   src_points[index].data[2] = z;
                } else {
                   index = -1;
                }
             }
             src_indices[idx] = index;
          }
       }
    }
    
    int TMP_SIZE = TGT_SIZE * sizeof(cuMat<float, 3>);
    cudaMalloc(reinterpret_cast<void**>(&d_model_points), TMP_SIZE);
    cudaMemcpy(d_model_points, model_points, TMP_SIZE, cudaMemcpyHostToDevice);

    int TIP_SIZE = SRC_SIZE * sizeof(int);
    cudaMalloc(reinterpret_cast<void**>(&d_model_indices), TIP_SIZE);
    cudaMemcpy(d_model_indices, model_indices, TIP_SIZE,
               cudaMemcpyHostToDevice);

    if (allocate_src) {
       int SMP_SIZE = SRC_POINT_SIZE * sizeof(cuMat<float, 3>);
       cudaMalloc(reinterpret_cast<void**>(&d_src_points), SMP_SIZE);
       cudaMemcpy(d_src_points, src_points, SMP_SIZE, cudaMemcpyHostToDevice);

       int SIP_SIZE = SRC_SIZE * sizeof(int);
       cudaMalloc(reinterpret_cast<void**>(&d_src_indices), SIP_SIZE);
       cudaMemcpy(d_src_indices, src_indices, SIP_SIZE,
                  cudaMemcpyHostToDevice);
    }

    Correspondence *d_correspondences;
    cudaMalloc(reinterpret_cast<void**>(&d_correspondences),
               sizeof(Correspondence) * image_size);
    cudaMemcpy(d_correspondences, correspondences,
               sizeof(Correspondence) * image_size, cudaMemcpyHostToDevice);
        
    dim3 block_size(cuDivUp(IMAGE_WIDTH, GRID_SIZE),
                    cuDivUp(IMAGE_HEIGHT, GRID_SIZE));
    dim3 grid_size(GRID_SIZE, GRID_SIZE);
    
    findCorrespondencesGPU<<<block_size, grid_size>>>(
       d_correspondences, d_src_points, d_src_indices, d_model_points,
       d_model_indices, IMAGE_WIDTH, IMAGE_HEIGHT, target_projection.height);

    // Correspondence *correspondences = reinterpret_cast<Correspondence*>(
    //    std::malloc(sizeof(Correspondence) * image_size));
    cudaMemcpy(correspondences, d_correspondences,
               sizeof(Correspondence) * image_size, cudaMemcpyDeviceToHost);

    // const float max_value = 15.0f;
    // const float min_value = -max_value;
    
    energy = 0.0f;
    int match_counter = 0;
    for (int i = 0; i < image_size; i++) {
       if ((correspondences[i].query_index > -1 &&
            correspondences[i].query_index < image_size) &&
           (correspondences[i].match_index > -1 &&
            correspondences[i].match_index < image_size)) {

          int model_index = correspondences[i].query_index;
          int src_index = correspondences[i].match_index;
          PointTYPE model_pt = target_points->points[model_index];
          PointTYPE src_pt = source_points->points[src_index];
          
          if (!isnan(model_pt.x) && !isnan(model_pt.y) && !isnan(model_pt.z) &&
              !isnan(src_pt.x) && !isnan(src_pt.y) && !isnan(src_pt.z)) {
             pcl::Correspondence c;
             c.index_query = correspondences[i].query_index;
             c.index_match = correspondences[i].match_index;
             corr.push_back(c);

             energy += correspondences[i].distance;
             match_counter++;
          }
       }
    }
    
    energy /= static_cast<float>(match_counter);
    energy = (match_counter == 0) ? -1.0f : energy;

    free(src_points);
    cudaFree(d_src_indices);
    cudaFree(d_src_points);
    
    free(correspondences);
    cudaFree(d_correspondences);
    cudaFree(d_model_points);
    cudaFree(d_model_indices);
    
    return true;
}



/**
 * DEBUG
 */

__global__ __forceinline__
void estimateCorrespondencesKernel(Correspondence *correspondences,
                                   float *d_model_points,
                                   int *d_target_indices,
                                   float *d_src_points,
                                   int *d_src_indices,
                                   const int image_size,
                                   const int step_size,
                                   const int wsize) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < image_size) {
       /* for points increment by 12 and 1 for indices */
       int model_index = d_target_indices[offset];
       
#ifdef _DEBUG
       if (model_index != -1) {
          printf("%d,  %d  %d \n", model_index, t_idx, t_idy);
       }
#endif
       if (model_index != -1) {
          cuRect rect;
          rect.x = t_idx - wsize/2;
          rect.y = t_idy - wsize/2;
          rect.width = wsize;
          rect.height = wsize;
          cuConditionROI(&rect, IMAGE_WIDTH, IMAGE_HEIGHT);

          int point_index = model_index * POINT_ELEMENTS;
          float model_pt[3];
          model_pt[0] = d_model_points[point_index];
          model_pt[1] = d_model_points[point_index + 1];
          model_pt[2] = d_model_points[point_index + 2];

#ifdef _DEBUG
          printf("INDEX: %d --- %3.4f,  %3.4f,  %3.4f\n",
                 model_index, model_pt[0], model_pt[1], model_pt[2]);
#endif

          float min_dsm = FLT_MAX;
          int min_ism = -1;
          for (int l = rect.y; l < rect.y + rect.height; l++) {
             for (int k = rect.x; k < rect.x + rect.width; k++) {
                int src_index = d_src_indices[k + (l * IMAGE_WIDTH)];
                if (src_index != -1) {
                   int spoint_index = src_index * POINT_ELEMENTS;
                   float src_pt[3];
                   src_pt[0] = d_src_points[spoint_index];
                   src_pt[1] = d_src_points[spoint_index + 1];
                   src_pt[2] = d_src_points[spoint_index + 2];
                   
                   // src_pt[0] = d_src_points[src_index].data[0];
                   // src_pt[1] = d_src_points[src_index].data[1];
                   // src_pt[2] = d_src_points[src_index].data[2];

                   float dist = cuEuclideanDistance(src_pt, model_pt, 3, false);
                   
                   if (dist >= 0.0f && dist < min_dsm && !isnan(dist)) {
                      min_dsm = dist;
                      min_ism = src_index;
                   }

#ifdef _DEBUG
                   if (model_index == 10) {
                      printf("SRC INDEX: %d --- %3.4f,  %3.4f,  %3.4f ---- %3.4f\n",
                             src_index, src_pt[0], src_pt[1], src_pt[2], dist);
                      printf("MATCH INDEX: %d --- %3.4f\n",
                             min_ism, min_dsm);
                   }
#endif
                }
             }
          }
          if (min_ism != -1 && min_dsm < DISTANCE_THRESH) {
             correspondences[model_index].query_index = model_index;
             correspondences[model_index].match_index = min_ism;
             correspondences[model_index].distance = min_dsm;
          } else {
             correspondences[model_index].query_index = -1;
             correspondences[model_index].match_index = -1;
          }

       }
    }
}


__global__ __forceinline__
void test_kernel(int *data, int image_size) {
   
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < image_size) {
       int index = data[offset];
       if (index != -1) {
          printf("%d,  %d  %d \n", index, t_idx, t_idy);
       }
    }
}


bool allocateCopyDataToGPU2(
    pcl::Correspondences &corr, float &energy,
    bool allocate_src,
    const pcl::PointCloud<PointTYPE>::Ptr source_points,
    const ProjectionMap &src_projection,
    const pcl::PointCloud<PointTYPE>::Ptr target_points,
    const ProjectionMap &target_projection) {
    if (source_points->empty() || target_points->empty()) {
       printf("\033[31m EMPTY POINTCLOUD FOR CORRESPONDENCES \033[0m\n");
       return false;
    }
    
    const int IN_TSIZE = static_cast<int>(target_points->size());
    const int IMG_SIZE = target_projection.indices.cols *
       target_projection.indices.rows;

    dim3 block_size(cuDivUp(IMAGE_WIDTH, GRID_SIZE),
                    cuDivUp(IMAGE_HEIGHT, GRID_SIZE));
    dim3 grid_size(GRID_SIZE, GRID_SIZE);
    
    //! copy model data
    float *d_target_points;
    cudaMalloc(reinterpret_cast<void**>(&d_target_points),
               IN_TSIZE * sizeof(float) * POINT_ELEMENTS);
    cudaMemcpy(d_target_points, target_points->points.data(),
               IN_TSIZE * sizeof(float) * POINT_ELEMENTS,
               cudaMemcpyHostToDevice);
    
    const int TARGET_BYTE = target_projection.indices.step *
       target_projection.indices.rows;
    int *d_target_indices;
    cudaMalloc(reinterpret_cast<void**>(&d_target_indices), TARGET_BYTE);
    // int *tdata = reinterpret_cast<int*>(target_projection.indices.data);
    cudaMemcpy(d_target_indices,
               reinterpret_cast<int*>(target_projection.indices.data),
               TARGET_BYTE, cudaMemcpyHostToDevice);

    float *d_src_points;
    int *d_src_indices;
    if (allocate_src) {
       // cudaFree(d_src_points);
       // cudaFree(d_src_indices);

       const int IN_SSIZE = static_cast<int>(source_points->size());
       cudaMalloc(reinterpret_cast<void**>(&d_src_points),
                  IN_SSIZE * sizeof(float) * POINT_ELEMENTS);
       cudaMemcpy(d_src_points, source_points->points.data(),
                  IN_SSIZE * sizeof(float) * POINT_ELEMENTS,
                  cudaMemcpyHostToDevice);
       
       const int SRC_BYTE = src_projection.indices.step *
          src_projection.indices.rows;
       cudaMalloc(reinterpret_cast<void**>(&d_src_indices), SRC_BYTE);
       cudaMemcpy(d_src_indices, reinterpret_cast<int*>(
                     src_projection.indices.data), SRC_BYTE,
                  cudaMemcpyHostToDevice);
    }

    Correspondence *d_correspondences;
    cudaMalloc(reinterpret_cast<void**>(&d_correspondences),
               sizeof(Correspondence) * IN_TSIZE);

    estimateCorrespondencesKernel<<<block_size, grid_size>>>(
       d_correspondences, d_target_points, d_target_indices,
       d_src_points, d_src_indices, IMG_SIZE,
       target_projection.indices.step, target_projection.height);
    
    // for (int i = 0; i < target_points->size(); i++) {
    //    std::cout << target_points->points[i]  << "\n";
    // }
    // return -1;

    Correspondence *correspondences = reinterpret_cast<Correspondence*>(
       std::malloc(sizeof(Correspondence) * IN_TSIZE));
    cudaMemcpy(correspondences, d_correspondences,
               sizeof(Correspondence) * IN_TSIZE, cudaMemcpyDeviceToHost);
    
    energy = 0.0f;
    int match_counter = 0;
    for (int i = 0; i < IN_TSIZE; i++) {
       if ((correspondences[i].query_index > -1 &&
            correspondences[i].query_index < IMG_SIZE) &&
           (correspondences[i].match_index > -1 &&
            correspondences[i].match_index < IMG_SIZE)) {

          int model_index = correspondences[i].query_index;
          int src_index = correspondences[i].match_index;
          PointTYPE model_pt = target_points->points[model_index];
          PointTYPE src_pt = source_points->points[src_index];
          
          if (!isnan(model_pt.x) && !isnan(model_pt.y) && !isnan(model_pt.z) &&
              !isnan(src_pt.x) && !isnan(src_pt.y) && !isnan(src_pt.z)) {
             pcl::Correspondence c;
             c.index_query = correspondences[i].query_index;
             c.index_match = correspondences[i].match_index;
             corr.push_back(c);

#ifdef _DEBUG
             std::cout << correspondences[i].query_index  << ", ";
             std::cout << correspondences[i].match_index  << ", ";
             std::cout << correspondences[i].distance  << "\n";
#endif
             energy += correspondences[i].distance;
             match_counter++;
          }
       }
    }
    
    energy /= static_cast<float>(match_counter);
    energy = (match_counter == 0) ? -1.0f : energy;

    std::cout << "\nENERGY: " << energy << "\t"  << match_counter  << "\n";

    cudaFree(d_src_indices);
    cudaFree(d_src_points);
    
    free(correspondences);
    cudaFree(d_correspondences);
    cudaFree(d_target_points);
    cudaFree(d_target_indices);
    
    return true;
}


void cudaGlobalAllocFree() {
    // cudaFree(d_src_indices);
    // cudaFree(d_src_points);
    // free(src_points);
}
