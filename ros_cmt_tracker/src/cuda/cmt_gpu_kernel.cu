
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


void processFrameGPU(cv::Mat im_gray) {
    if (im_gray.empty()) {
       printf("\033[31m EMPTY IMAGE FOR TRACKING \033[0m]\n");
       return;
    }
    
}
