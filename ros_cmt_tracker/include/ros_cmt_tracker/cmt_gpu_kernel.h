
#ifndef _CMT_GPU_KERNEL_H_
#define _CMT_GPU_KERNEL_H_

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <opencv2/opencv.hpp>

void sortConfPairs(int &, int &, const std::vector<float>);

void processFrameGPU(cv::Mat);



#endif /* _CMT_GPU_KERNEL_H_ */

