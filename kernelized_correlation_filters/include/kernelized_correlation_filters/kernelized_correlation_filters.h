
#ifndef _KERNELIZED_CORRELATION_FILTERS_H_
#define _KERNELIZED_CORRELATION_FILTERS_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PolygonStamped.h>
#include <jsk_recognition_msgs/Rect.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include<kernelized_correlation_filters/kcf.h>
#include <kernelized_correlation_filters/deep_feature_extraction.h>

#include <omp.h>
#include <cmath>

class KernelizedCorrelationFilters {

 private:
    cv::Rect_<int> screen_rect_;
    bool tracker_init_;
    int width_;
    int height_;
    int block_size_;
    int downsize_;
    cv::Mat prev_frame_;

    boost::shared_ptr<KCF_Tracker> tracker_;

 protected:
    void onInit();
    void subscribe();
    void unsubscribe();

    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_image_;
    ros::Subscriber sub_screen_pt_;
    ros::Publisher pub_image_;
    unsigned int threads_;
   
 public:
    KernelizedCorrelationFilters();
    void imageCB(
       const sensor_msgs::Image::ConstPtr &);
    void screenPtCB(
       const geometry_msgs::PolygonStamped &);
   
};


#endif /* _KERNELIZED_CORRELATION_FILTERS_H_ */
