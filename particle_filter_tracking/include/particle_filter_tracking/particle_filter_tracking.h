// Copyright (C) 2015 by Krishneel Chaudhary,
// JSK Lab, The University of Tokyo

#ifndef _PARTICLE_FILTER_TRACKING_H_
#define _PARTICLE_FILTER_TRACKING_H_

#include <particle_filter_tracking/particle_filter.h>
#include <particle_filter_tracking/color_histogram.h>
#include <particle_filter_tracking/histogram_of_oriented_gradients.h>

#include <ros/ros.h>
#include <ros/console.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

#include <geometry_msgs/PolygonStamped.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/thread/mutex.hpp>

class ParticleFilterTracking: public ParticleFilter,
                              public HOGFeatureDescriptor,
                              public ColorHistogram {

    struct Features {
       std::vector<cv::Mat> color;
       std::vector<cv::Mat> hog;
    };
  
 private:
    virtual void onInit();
    virtual void subscribe();
    virtual void unsubscribe();

    cv::Rect_<int> screen_rect_;

    std::vector<cv::Mat> reference_object_histogram_;
    std::vector<cv::Mat> reference_background_histogram_;

    int width_;
    int height_;
    int block_size_;
    int downsize_;
   
    int hbins;
    int sbins;
   
    cv::Mat dynamics;
    std::vector<Particle> particles;
    cv::RNG randomNum;
    std::vector<cv::Point2f> particle_prev_position;
    cv::Mat prevFrame;

    bool tracker_init_;
   
 public:
    ParticleFilterTracking();
    virtual void imageCallback(
       const sensor_msgs::Image::ConstPtr &);
    virtual void screenPointCallback(
       const geometry_msgs::PolygonStamped &);

    void initializeTracker(
       const cv::Mat &, cv::Rect &);
    void runObjectTracker(
       cv::Mat *image, cv::Rect &rect);
   
    std::vector<cv::Mat> imagePatchHistogram(
       cv::Mat &);
    std::vector<cv::Mat> particleHistogram(
      cv::Mat &, std::vector<Particle> &);
    std::vector<double> colorHistogramLikelihood(
       std::vector<cv::Mat> &);
    void roiCondition(cv::Rect &, cv::Size);
   
 protected:
    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_image_;
    ros::Subscriber sub_screen_pt_;
    ros::Publisher pub_image_;
    unsigned int threads_;
};

#endif  // _PARTICLE_FILTER_TRACKING_H_
