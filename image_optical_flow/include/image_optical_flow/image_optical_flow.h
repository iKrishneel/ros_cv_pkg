#ifndef _IMAGE_OPTICAL_FLOW_H_
#define _IMAGE_OPTICAL_FLOW_H_

// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo

#include <ros/ros.h>
#include <ros/console.h>

#include <opencv2/opencv.hpp>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <boost/thread/mutex.hpp>

class ImageOpticalFlow {

 private:
    virtual void onInit();
    virtual void subscribe();
    virtual void unsubscribe();

    cv::Mat prev_image_;
    std::vector<cv::Point2f> prev_pts_;
    int icounter;
   
 public:
    ImageOpticalFlow();
    virtual void imageCallback(
       const sensor_msgs::Image::ConstPtr &);
    virtual void drawFlowField(
       cv::Mat &, cv::Mat &,
       std::vector<cv::Point2f> &,
       std::vector<cv::Point2f> &);
    void computeOpticalFlow(
       cv::Mat &, cv::Mat &,
       std::vector<cv::Point2f> &,
       std::vector<cv::Point2f> &);
 protected:
    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_image_;
    ros::Publisher pub_image_;
};


#endif  // _IMAGE_OPTICAL_FLOW_H_
