
#ifndef _OBJECT_DETECTOR_TRAINER_H_
#define _OBJECT_DETECTOR_TRAINER_H_

#include <object_recognition/histogram_of_oriented_gradients.h>
#include <object_recognition/local_binary_patterns.h>

#include <ros/ros.h>
#include <ros/console.h>

#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <fstream>

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
#define CLEAR "\033[2J"  // clear screen escape code

class ObjectDetectorTrainer:  public HOGFeatureDescriptor,
                              public LocalBinaryPatterns {
 private:
    int swindow_x_;
    int swindow_y_;
    std::string dataset_path_;

    std::string object_dataset_filename_;
    std::string nonobject_dataset_filename_;
    std::string trained_classifier_name_;
   
    boost::shared_ptr<cv::SVM> supportVectorMachine_;
    void writeTrainingManifestToDirectory(cv::FileStorage &);

 public:
    ObjectDetectorTrainer();
    virtual void trainObjectClassifier(
       std::string, std::string);
    virtual void readDataset(
       std::string, cv::Mat &,
       cv::Mat &, bool = false, const int = 0);
    virtual void extractFeatures(
       cv::Mat &, cv::Mat &);
    virtual void trainBinaryClassSVM(
       const cv::Mat &, const cv::Mat &);
    virtual void concatenateCVMat(
       const cv::Mat &, const cv::Mat &, cv::Mat &, bool = true);
};


#endif  // _OBJECT_DETECTOR_TRAINER_H_
