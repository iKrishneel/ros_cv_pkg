
#ifndef _CMT_GPU_KERNEL_H_
#define _CMT_GPU_KERNEL_H_

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

#define GRID_SIZE 16

bool forwardBackwardError(float *,
                          const std::vector<cv::Point2f> &,
                          const std::vector<cv::Point2f> &);
bool forwardBackwardError(float *,
                          const std::vector<cv::Point2f> &,
                          cv::cuda::GpuMat);

void processFrameGPU(cv::Mat);



#endif /* _CMT_GPU_KERNEL_H_ */

