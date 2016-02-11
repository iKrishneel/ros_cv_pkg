
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
#include <std_msgs/Header.h>

class PointCloudImageCreator {

 public:
    typedef pcl::PointXYZRGB PointT;

    PointCloudImageCreator();
    virtual void cloudCallback(
       const sensor_msgs::PointCloud2::ConstPtr &);
    virtual void imageCallback(
      const sensor_msgs::Image::ConstPtr &);
    virtual void cameraInfoCallback(
       const sensor_msgs::CameraInfo::ConstPtr &);
    cv::Mat projectPointCloudToImagePlane(
       const pcl::PointCloud<PointT>::Ptr,
       const sensor_msgs::CameraInfo::ConstPtr &, cv::Mat &);
    cv::Mat interpolateImage(const cv::Mat &, const cv::Mat &);
    void cvMorphologicalOperations(const cv::Mat &, cv::Mat &, bool);
    cv::Rect createMaskImages(const cv::Mat &, cv::Mat &, cv::Mat &);

 protected:
    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_cam_info_;
    ros::Subscriber sub_cloud_;
    ros::Subscriber sub_image_;
    ros::Publisher pub_image_;
    ros::Publisher pub_iimage_;
    ros::Publisher pub_cloud_;
    ros::Publisher pub_fmask_;
    ros::Publisher pub_bmask_;
   
    virtual void onInit();
    virtual void subsribe();

 private:
    sensor_msgs::CameraInfo::ConstPtr camera_info_;
    bool is_mask_image_;
    bool is_roi_;
    std_msgs::Header header_;
    cv::Mat foreground_mask_;
    cv::Rect rect_;
    boost::mutex mutex_;

    bool is_info_;
};
