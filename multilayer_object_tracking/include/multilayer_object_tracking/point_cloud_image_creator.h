
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

 public:
    typedef pcl::PointXYZRGB PointT;

    PointCloudImageCreator();

    virtual void cloudCallback(
       const sensor_msgs::PointCloud2::ConstPtr &);
    virtual void cameraInfoCallback(
       const sensor_msgs::CameraInfo::ConstPtr &);
    virtual void imageCallback(
       const sensor_msgs::Image::ConstPtr &);
   
    cv::Mat projectPointCloudToImagePlane(
       const pcl::PointCloud<PointT>::Ptr,
       const sensor_msgs::CameraInfo::ConstPtr &, cv::Mat &);
    cv::Mat interpolateImage(const cv::Mat &, const cv::Mat &);

    void cvMorphologicalOperations(const cv::Mat &, cv::Mat &, bool);
    void contourSmoothing(const cv::Mat &, cv::Mat &, cv::Rect_<int> &);
   
    virtual void onInit();
    virtual void subsribe();

 private:
    cv::Mat image_;
   
 protected:
    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_cam_info_;
    ros::Subscriber sub_cloud_;
    ros::Subscriber sub_image_;
   
    ros::Publisher pub_image_;
    // ros::Publisher pub_iimage_;
    ros::Publisher pub_cloud_;

    sensor_msgs::CameraInfo::ConstPtr camera_info_;
};
