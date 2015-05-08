
#ifndef _IMAGE_ANNOTATION_TOOL_H_
#define _IMAGE_ANNOTATION_TOOL_H_

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PolygonStamped.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <boost/filesystem.hpp>
#include <boost/date_time/local_time/local_time.hpp>

#include<fstream>
#include <dirent.h>

class ImageAnnotationTool {

 private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    ros::Subscriber screenpt_sub_;
   
    cv::Mat fore_histogram_;
    cv::Mat back_histogram_;
    int iteration_counter_;
    cv::Mat template_image;
    bool mark_model_;
    cv::Rect_<int> obj_rect_;
    
    // parameter from launch
    std::string run_type_;
    std::string save_folder_name_;
    bool background_mask_;
    int swindow_x_;
    int swindow_y_;
   
    boost::shared_ptr<rosbag::Bag> dataset_bag_;
   
 public:
    explicit ImageAnnotationTool(const std::string);
    void imageCb(
       const sensor_msgs::ImageConstPtr&);
    void screenRectangleCb(
        const geometry_msgs::PolygonStamped::Ptr &);
    cv::Mat processFrame(
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

    bool directoryExists(const char*);
    bool saveDirectoryHandler(
        const std::string, std::string &);
    template<class T>
    std::string timeToStr(T);
    
 protected:
    void subscribe();
   
};

#endif  // _IMAGE_ANNOTATION_TOOL_H_
