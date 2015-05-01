#ifndef _LOCAL_BINARY_PATTERNS_H_
#define _LOCAL_BINARY_PATTERNS_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <opencv2/opencv.hpp>

class LocalBinaryPatterns {

 private:
    template <typename _Tp>
    void localBinaryPatterns(
       const cv::Mat &, cv::Mat &);
    virtual void patchWiseLBP(
       const cv::Mat &, cv::Mat &, const cv::Size, const int, bool = false);
    virtual cv::Mat histogramLBP(
       const cv::Mat &, int = 0, int = 255, bool = true);
    virtual void getLBP(
       const cv::Mat &, cv::Mat &);
   
 public:
    LocalBinaryPatterns();
    virtual cv::Mat computeLBP(
       const cv::Mat &, const cv::Size = cv::Size(16, 16),
       const int = 8, bool = false, bool = false);
};


#endif  // _LOCAL_BINARY_PATTERNS_H_
