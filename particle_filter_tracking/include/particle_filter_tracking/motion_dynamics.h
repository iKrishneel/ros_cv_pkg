//  Created by Chaudhary Krishneel, JSK Lab, The University of Tokyo
//  Copyright (c) 2015 Chaudhary Krishneel. All rights reserved.

#ifndef _MOTION_DYNAMICS_H_
#define _MOTION_DYNAMICS_H_

#include <opencv2/opencv.hpp>
#include <vector>

class MotionDynamics {
 private:
    void buildImagePyramid(
       const cv::Mat &, std::vector<cv::Mat> &);
    const int feature_size;
 public:
    MotionDynamics();
    void getOpticalFlow(
       const cv::Mat &, const cv::Mat &,
       std::vector<cv::Point2f> &, bool CV_DEFAULT(false));
    
};
#endif  //_MOTION_DYNAMICS_H_
