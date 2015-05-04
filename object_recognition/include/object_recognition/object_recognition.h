
#ifndef _OBJECT_RECOGNITION_H_
#define _OBJECT_RECOGNITION_H_

#include <object_recognition/histogram_of_oriented_gradients.h>
#include <object_recognition/local_binary_patterns.h>
#include <object_recognition/svm.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/RectArray.h>
#include <jsk_pcl_ros/pcl_conversion_util.h>

#include <dynamic_reconfigure/server.h>
#include <object_recognition/ObjectDetectionConfig.h>

#include <pcl/point_types.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

#include <map>
#include <vector>
#include <string>

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

class ObjectRecognition: public HOGFeatureDescriptor,
                         public LocalBinaryPatterns {
 private:

    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    ros::Publisher pub_indices_;
    ros::Publisher pub_rects_;
   
    ros::ServiceClient nms_client_;
   
   // cv::Size swindow_;
    boost::shared_ptr<cv::SVM> supportVectorMachine_;
    void concatenateCVMat(
       const cv::Mat &, const cv::Mat &, cv::Mat &, bool = true);
   
 public:
    explicit ObjectRecognition(const std::string);
    virtual void trainObjectClassifier();
    virtual void readDataset(
       std::string, std::vector<cv::Mat> &,
       cv::Mat &, bool = false, const int = 0);
    virtual void extractFeatures(
       const std::vector<cv::Mat> &, cv::Mat &);
    virtual void trainBinaryClassSVM(
       const cv::Mat &, const cv::Mat &);
    virtual void imageCb(
       const sensor_msgs::ImageConstPtr&);

    virtual std::vector<cv::Rect_<int> > runObjectRecognizer(
      const cv::Mat &, const cv::Size, const float, const int, const int);
    virtual void objectRecognizer(
       const cv::Mat &, std::multimap<float, cv::Rect_<int> > &,
       const cv::Size, const int = 16);
    virtual void pyramidialScaling(
       cv::Size &, const float);
    virtual std::vector<cv::Rect_<int> > nonMaximumSuppression(
       std::multimap<float, cv::Rect_<int> > &, const float); 
   void convertCvRectToJSKRectArray(
       const std::vector<cv::Rect_<int> > &,
      jsk_recognition_msgs::RectArray &, const int, const cv::Size);
    void objectBoundingBoxPointCloudIndices(
       const std::vector<cv::Rect_<int> > &,
       std::vector<pcl::PointIndices> &,
       const int,
       const cv::Size);
   
    virtual void subscribe();
    virtual void configCallback(
        object_recognition::ObjectDetectionConfig &, uint32_t);

    struct svm_problem convertCvMatToLibSVM(
        const cv::Mat &feature_mat, const cv::Mat &,
        const svm_parameter &);
    
 protected:
    boost::mutex mutex_;
    float scale_;
    int stack_size_;
    int incrementor_;
    int downsize_;
    std::string model_name_;
    std::string dataset_path_;
    int swindow_x;
    int swindow_y;

    dynamic_reconfigure::Server<
       object_recognition::ObjectDetectionConfig>  server;
   
};

#endif  // _OBJECT_RECOGNITION_H_
