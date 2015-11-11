
#ifndef _SALIENCY_MAP_GENERATOR_H_
#define _SALIENCY_MAP_GENERATOR_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <opencv2/opencv.hpp>

class SaliencyMapGenerator {

 public:
  SaliencyMapGenerator();
  bool computeSaliencyImpl(cv::Mat, cv::Mat &);

 private:
  void calcIntensityChannel(cv::Mat, cv::Mat);
  void copyImage(cv::Mat, cv::Mat);
  void getIntensityScaled(cv::Mat, cv::Mat, cv::Mat, cv::Mat, int);
  float getMean(cv::Mat, cv::Point2i, int, int);
  void mixScales(cv::Mat *, cv::Mat, cv::Mat *, cv::Mat, const int);
  void mixOnOff(cv::Mat intensityOn, cv::Mat intensityOff, cv::Mat intensity);
  void getIntensity(cv::Mat, cv::Mat, cv::Mat, cv::Mat, bool);
};

#endif	 // _SALIENCY_MAP_GENERATOR_H_


