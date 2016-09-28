
#ifndef _CMT_GPU_KERNEL_H_
#define _CMT_GPU_KERNEL_H_

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

#define CUDART_PI_F 3.141592654f
#define GRID_SIZE 16

bool forwardBackwardError(float *,
                          const std::vector<cv::Point2f> &,
                          const std::vector<cv::Point2f> &);
bool forwardBackwardError(float *,
                          const std::vector<cv::Point2f> &,
                          cv::cuda::GpuMat);

bool keypointsAngluarDifferenceGPU(float *, float *, float *,
                                   std::vector<int>, std::vector<int>,
                                   const cv::Mat);


#endif /* _CMT_GPU_KERNEL_H_ */

