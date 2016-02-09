#ifndef _IMAGE_OPTICAL_FLOW_H_
#define _IMAGE_OPTICAL_FLOW_H_

// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo

#include <ros/ros.h>
#include <ros/console.h>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/videostab/videostab.hpp>
#include <opencv2/videostab/global_motion.hpp>
#include <opencv2/videostab/optical_flow.hpp>
#include <opencv2/video/video.hpp>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <boost/thread/mutex.hpp>

class ImageOpticalFlow {

    struct FeatureInfo {
       cv::Mat image;
       cv::Mat descriptor;
       std::vector<cv::KeyPoint> keypoints;
    };
   
 private:
    virtual void onInit();
    virtual void subscribe();
    virtual void unsubscribe();

    cv::Mat prev_image_;
    std::vector<cv::Point2f> prev_pts_;
    int icounter;

    FeatureInfo prev_info_;
    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorExtractor> descriptor_;
   
 public:
    ImageOpticalFlow();
    virtual void imageCallback(
       const sensor_msgs::Image::ConstPtr &);
   
    void buildImagePyramid(
       const cv::Mat &, std::vector<cv::Mat> &);
    void getOpticalFlow(
      const cv::Mat &, const cv::Mat &,
      std::vector<cv::Point2f> &nextPts,
      std::vector<cv::Point2f> &prevPts,
      std::vector<uchar> &status);

    void computeOpticalFlow(
       cv::Mat &prev_frame, cv::Mat &cur_frame,
       std::vector<cv::Point2f> &prev_pts,
       std::vector<cv::Point2f> &next_pts);
   
    void drawFlowField(
       cv::Mat &frame, cv::Mat &prev_frame,
       std::vector<cv::Point2f> &next_pts,
       std::vector<cv::Point2f> &prev_pts);
   
   
    void forwardBackwardMatchingAndFeatureCorrespondance(
       const cv::Mat, const cv::Mat, FeatureInfo &);
   
 protected:
    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_image_;
    ros::Publisher pub_image_;
};


#endif  // _IMAGE_OPTICAL_FLOW_H_
