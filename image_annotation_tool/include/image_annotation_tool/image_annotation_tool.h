
#ifndef _IMAGE_ANNOTATION_TOOL_H_
#define _IMAGE_ANNOTATION_TOOL_H_

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include<fstream>

class ImageAnnotationTool {

 private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
   
    cv::Mat fore_histogram_;
    cv::Mat back_histogram_;
    int iteration_counter_;
    int model_save_counter_;
    std::string templ_path_;
    std::string model_save_path_;
    std::ofstream out_file_;
   
 public:
    ImageAnnotationTool(std::string);
    void imageCb(
       const sensor_msgs::ImageConstPtr&);
    void processFrame(
       const cv::Mat &,
       const cv::Mat &);
    cv::Mat shapeProbabilityMap(
       const cv::Mat &);
    cv::Mat pixelWiseClassification(
       const cv::Mat &);
    void computeHistogram(
       const cv::Mat &,
       cv::Mat &,
       bool = true);
    cv::Mat cvTemplateMatching(
       const cv::Mat &,
       const cv::Mat &,
       const int = CV_TM_CCOEFF);
    void constructColorModel(
       const cv::Mat &,
       const cv::Rect &,
       bool = false);
    void cvMorphologicalOperation(
       cv::Mat &,
       const int = 5,
       bool = false);
    cv::Mat segmentationProbabilityMap(
       cv::Mat &,
       cv::Mat &);

    cv::Mat graphCutSegmentation(
       const cv::Mat &,
       cv::Mat &,
       cv::Rect_<int> &);
    cv::Mat createMaskImage(
       cv::Mat &);
    cv::Rect_<int> modelResize(
       const cv::Mat &);
    template<typename T>
    std::string convertNumber2String(T);
    void saveImageModel(
       const cv::Mat &,
       const std::string &,
       bool = true);
    void bootStrapNonObjectDataset(
       const cv::Mat &,
       const cv::Size,
       const int = 16);
   
   
 protected:
    void subscribe();
   
};


#endif  // _IMAGE_ANNOTATION_TOOL_H_
