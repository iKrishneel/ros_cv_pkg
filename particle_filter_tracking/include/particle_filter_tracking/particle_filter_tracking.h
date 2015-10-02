// Copyright (C) 2015 by Krishneel Chaudhary,
// JSK Lab, The University of Tokyo

#ifndef _PARTICLE_FILTER_TRACKING_H_
#define _PARTICLE_FILTER_TRACKING_H_

#include <particle_filter_tracking/particle_filter.h>
#include <particle_filter_tracking/motion_dynamics.h>
#include <particle_filter_tracking/color_histogram.h>

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

#include <geometry_msgs/PolygonStamped.h>
#include <vector>

class ParticleFilterTracking: public ParticleFilter,
                              public MotionDynamics,
                              public ColorHistogram {

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
    std::vector<cv::Point2f> prevPts;
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
    std::vector<double> motionLikelihood(
       std::vector<double> &, std::vector<Particle> &,
       std::vector<Particle> &);
    cv::Point2f motionCovarianceEstimator(
       std::vector<cv::Point2f> &, std::vector<Particle> &);
    double motionVelocityLikelihood(
       double);
    double gaussianNoise(double, double);
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
