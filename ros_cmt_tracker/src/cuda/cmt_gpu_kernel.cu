
#include <ros_cmt_tracker/cmt_gpu_kernel.h>

void processFrameGPU(cv::Mat im_gray) {
    if (im_gray.empty()) {
       printf("\033[31m EMPTY IMAGE FOR TRACKING \033[0m]\n");
       return;
    }
    
}
