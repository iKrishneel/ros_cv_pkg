
#ifndef _GRAPH_CUT_SEGMENTATION_
#define _GRAPH_CUT_SEGMENTATION_

#include <ros/ros.h>
#include <ros/console.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

class GraphCutSegmentation {
 private:
    cv::Rect model_resize(cv::Mat &);
    cv::Mat createMaskImage(cv::Mat &);
    
 public:
    cv::Mat graphCutSegmentation(
       cv::Mat &, cv::Mat &, cv::Rect &, int = 1);
};

#endif  // _GRAPH_CUT_SEGEMENTATION_
