
#ifndef _CMT_GPU_KERNEL_H_
#define _CMT_GPU_KERNEL_H_

#include <opencv2/opencv.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/features2d/features2d.hpp>

void processFrameGPU(cv::Mat);


#endif /* _CMT_GPU_KERNEL_H_ */

