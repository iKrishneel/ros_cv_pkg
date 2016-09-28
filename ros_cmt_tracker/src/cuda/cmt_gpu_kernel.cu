
#include <ros_cmt_tracker/cmt_gpu_kernel.h>

#define CUDA_ERROR_CHECK(process) {               \
      cudaAssert((process), __FILE__, __LINE__);  \
   }                                              \
      
void cudaAssert(cudaError_t code, char *file, int line, bool abort) {
    if (code != cudaSuccess) {
       fprintf(stderr, "GPUassert: %s %s %dn",
               cudaGetErrorString(code), file, line);
       if (abort) {
         exit(code);
      }
    }
}


__host__ __device__ __align__(16) int
    cuDivUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


/*****************************************************
 * COMPUTING MOTION ERROR DISTANCE
 *****************************************************/

__global__ __forceinline__
void fbErrorKernel(float *d_error, const float *d_pt_data,
                   const float *d_ptb_data, int SIZE) {
    int t_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (t_id < SIZE) {
       int index = t_id * 2;
       float val = powf(d_ptb_data[index] - d_pt_data[index], 2) +
          powf(d_ptb_data[index + 1] - d_pt_data[index + 1], 2);
       d_error[t_id] = sqrtf(val);
    }
}

bool forwardBackwardError(float *errors,
                          const std::vector<cv::Point2f> &pts,
                          const std::vector<cv::Point2f> &pts_back) {
    if (pts.empty() || pts_back.empty()) {
       printf("\033[31m ERROR[forwardBackwardError]: EMPTY \033[0m\n");
       return false;
    }
    const size_t PTS_SIZE = sizeof(float) * static_cast<int>(pts.size()) * 2;
    float *d_pt_data;
    cudaMalloc(reinterpret_cast<void**>(&d_pt_data), PTS_SIZE);
    cudaMemcpy(d_pt_data, pts.data(), PTS_SIZE, cudaMemcpyHostToDevice);
    
    const size_t PTSB_SIZE = sizeof(float) *
       static_cast<int>(pts_back.size()) * 2;
    float *d_ptb_data;
    cudaMalloc(reinterpret_cast<void**>(&d_ptb_data), PTSB_SIZE);
    cudaMemcpy(d_ptb_data, pts_back.data(), PTSB_SIZE, cudaMemcpyHostToDevice);
    
    float *d_error;
    cudaMalloc(reinterpret_cast<void**>(&d_error), PTS_SIZE/2);
    cudaMemset(d_error, 0.0f, PTS_SIZE/2);

    const int SIZE = static_cast<int>(pts.size());
    fbErrorKernel<<<SIZE, 1>>>(d_error, d_pt_data, d_ptb_data, SIZE);

    // float *error = reinterpret_cast<float*>(std::malloc(PTS_SIZE/2));
    cudaMemcpy(errors, d_error, PTS_SIZE/2, cudaMemcpyDeviceToHost);
    // std::memcpy(errors, error, PTS_SIZE/2);
    

    cudaFree(d_error);
    cudaFree(d_pt_data);
    cudaFree(d_ptb_data);
    return true;
}

bool forwardBackwardError(float *errors,
                          const std::vector<cv::Point2f> &pts,
                          cv::cuda::GpuMat d_pts_back) {
    if (pts.empty() || d_pts_back.empty()) {
       printf("\033[31m ERROR[forwardBackwardError]: EMPTY \033[0m\n");
       return false;
    }
    
    const size_t PTS_SIZE = sizeof(float) * static_cast<int>(pts.size()) * 2;
    float *d_pt_data;
    cudaMalloc(reinterpret_cast<void**>(&d_pt_data), PTS_SIZE);
    cudaMemcpy(d_pt_data, pts.data(), PTS_SIZE, cudaMemcpyHostToDevice);

    const size_t PTSB_SIZE = d_pts_back.step * d_pts_back.rows;
    float *d_ptb_data;
    cudaMalloc(reinterpret_cast<void**>(&d_ptb_data), PTSB_SIZE);
    // cudaMemcpy(d_ptb_data, d_pts_back.data, PTSB_SIZE,
    //            cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptb_data, d_pts_back.data, PTSB_SIZE,
               cudaMemcpyDeviceToDevice);
    
    float *d_error;
    cudaMalloc(reinterpret_cast<void**>(&d_error), PTS_SIZE/2);
    cudaMemset(d_error, 0.0f, PTS_SIZE/2);

    const int SIZE = static_cast<int>(pts.size());
    fbErrorKernel<<<SIZE, 1>>>(d_error, d_pt_data, d_ptb_data, SIZE);
    cudaMemcpy(errors, d_error, PTS_SIZE/2, cudaMemcpyDeviceToHost);

    /*
      cudaMemcpy(errors, d_ptb_data, PTSB_SIZE, cudaMemcpyDeviceToHost);
    std::cout << PTSB_SIZE  << "\n";
    for (int i = 0; i < PTSB_SIZE/sizeof(float); i+=2) {
       std::cout << errors[i] << " " << errors[i+1] << "\n";
    }
    */
    cudaFree(d_error);
    cudaFree(d_pt_data);
    cudaFree(d_ptb_data);

    return true;
}

/*****************************************************
 * COMPUTING ANGLE DIFFERENCE
 *****************************************************/

template <typename T>
__device__ __forceinline__
T cuSign(T t) {
    if (t == 0) {
       return T(0);
    } else {
       return (t < 0) ? T(-1) : T(1);
    }
}

__global__ __forceinline__
void angleKernel(float *d_angles, float *d_orig_angles,
                 int *d_class_ind1, int *d_class_ind2,
                 float *d_pts_ind1, float *d_pts_ind2,
                 const int SIZE, const int LENGHT) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < SIZE) {
       
       int index = offset * 2;
       int idx = d_class_ind2[offset] + (d_class_ind1[offset] * LENGHT);
       
       float ang_diff = atan2(d_pts_ind2[index + 1] - d_pts_ind1[index + 1],
                              d_pts_ind2[index] - d_pts_ind1[index]);
       ang_diff -= d_orig_angles[idx];
       if (fabsf(ang_diff) > CUDART_PI_F) {
          ang_diff -= cuSign(ang_diff) * 2.0f * CUDART_PI_F;
       }
       d_angles[offset] = ang_diff;
    }
}

bool keypointsAngluarDifferenceGPU(
    float *angle_diffs,
    float *pts_ind1, float *pts_ind2,
    std::vector<int> class_ind1,
    std::vector<int> class_ind2,
    const cv::Mat angles) {

    const int MEM_SIZE = static_cast<int>(class_ind1.size());
    int *d_class_ind1;
    int BYTE = MEM_SIZE * sizeof(int);
    cudaMalloc(reinterpret_cast<void**>(&d_class_ind1), BYTE);
    cudaMemcpy(d_class_ind1, class_ind1.data(), BYTE, cudaMemcpyHostToDevice);

    int *d_class_ind2;
    cudaMalloc(reinterpret_cast<void**>(&d_class_ind2), BYTE);
    cudaMemcpy(d_class_ind2, class_ind2.data(), BYTE, cudaMemcpyHostToDevice);
    
    BYTE = MEM_SIZE * sizeof(float);
    float *d_pts_ind1;
    cudaMalloc(reinterpret_cast<void**>(&d_pts_ind1), BYTE * 2);
    cudaMemcpy(d_pts_ind1, pts_ind1, BYTE * 2, cudaMemcpyHostToDevice);

    float *d_pts_ind2;
    cudaMalloc(reinterpret_cast<void**>(&d_pts_ind2), BYTE * 2);
    cudaMemcpy(d_pts_ind2, pts_ind2, BYTE * 2, cudaMemcpyHostToDevice);

    float *d_angles;
    cudaMalloc(reinterpret_cast<void**>(&d_angles), BYTE);
    cudaMemset(d_angles, 0.0f, BYTE);
    
    float *d_orig_angles;
    BYTE = angles.step * angles.rows;
    cudaMalloc(reinterpret_cast<void**>(&d_orig_angles), BYTE);
    cudaMemcpy(d_orig_angles, angles.data, BYTE, cudaMemcpyHostToDevice);

    dim3 block_size(cuDivUp(MEM_SIZE/2 + 1, GRID_SIZE),
                    cuDivUp(MEM_SIZE/2 + 1, GRID_SIZE));
    dim3 grid_size(GRID_SIZE, GRID_SIZE);

    
    angleKernel<<<block_size, grid_size>>>(d_angles, d_orig_angles,
                                           d_class_ind1, d_class_ind2,
                                           d_pts_ind1, d_pts_ind2,
                                           MEM_SIZE, angles.cols);
    BYTE = MEM_SIZE * sizeof(float);
    cudaMemcpy(angle_diffs, d_angles, BYTE, cudaMemcpyDeviceToHost);
    
    cudaFree(d_class_ind1);
    cudaFree(d_class_ind2);
    cudaFree(d_pts_ind1);
    cudaFree(d_pts_ind2);
    cudaFree(d_angles);
    cudaFree(d_orig_angles);
}


void deepNestedVector(
    std::vector<std::vector<float> > data) {
   
}
