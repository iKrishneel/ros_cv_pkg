
#ifndef _SALIENCY_MAP_GENERATOR_H_
#define _SALIENCY_MAP_GENERATOR_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <opencv2/opencv.hpp>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <boost/thread/mutex.hpp>

#include <omp.h>

class SaliencyMapGenerator {

 public:
    explicit SaliencyMapGenerator(int = 1);
    bool computeSaliencyImpl(cv::Mat, cv::Mat &);
    void setNumThreads(int);
    void callback(const sensor_msgs::Image::ConstPtr &);

 private:
  void calcIntensityChannel(cv::Mat, cv::Mat);
  void copyImage(cv::Mat, cv::Mat);
  void getIntensityScaled(cv::Mat, cv::Mat, cv::Mat, cv::Mat, int);
  float getMean(cv::Mat, cv::Point2i, int, int);
  void mixScales(cv::Mat *, cv::Mat, cv::Mat *, cv::Mat, const int);
  void mixOnOff(cv::Mat intensityOn, cv::Mat intensityOff, cv::Mat intensity);
  void getIntensity(cv::Mat, cv::Mat, cv::Mat, cv::Mat, bool);

    int num_threads_;

    void onInit();
    void subscribe();

 protected:
    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_image_;
    ros::Publisher pub_image_;
};

#endif	 // _SALIENCY_MAP_GENERATOR_H_


