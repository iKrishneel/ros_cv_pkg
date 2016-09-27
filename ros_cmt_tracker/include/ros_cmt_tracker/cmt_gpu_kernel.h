
#ifndef _CMT_GPU_KERNEL_H_
#define _CMT_GPU_KERNEL_H_

#include <opencv2/opencv.hpp>

#define GRID_SIZE 16

bool forwardBackwardError(float *,
                          const std::vector<cv::Point2f> &,
                          const std::vector<cv::Point2f> &);
void processFrameGPU(cv::Mat);



#endif /* _CMT_GPU_KERNEL_H_ */

