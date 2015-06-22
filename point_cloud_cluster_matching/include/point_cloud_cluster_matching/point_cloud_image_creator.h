
#include <ros/ros.h>
#include <ros/console.h>

#include <opencv2/opencv.hpp>

#include <boost/thread/mutex.hpp>

#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class PointCloudImageCreator {

 private:
    typedef pcl::PointXYZRGB PointT;

 public:
   
    PointCloudImageCreator();
    cv::Mat projectPointCloudToImagePlane(
       const pcl::PointCloud<PointT>::Ptr,
       const sensor_msgs::CameraInfo::ConstPtr &, cv::Mat &);
    cv::Mat interpolateImage(const cv::Mat &, const cv::Mat &);
    void cvMorphologicalOperations(const cv::Mat &, cv::Mat &, bool);
    
    
 protected:
    boost::mutex lock_;
    sensor_msgs::CameraInfo::ConstPtr camera_info_;
};
