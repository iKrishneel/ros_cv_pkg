
#ifndef _PCL_FILTER_UTILS_H_
#define _PCL_FILTER_UTILS_H_

#include <dynamic_reconfigure/server.h>
#include <pcl_filter_utils/PointCloudFilterUtilsConfig.h>

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
#include <pcl/filters/passthrough.h>

class PointCloudFilterUtils {
    
 public:
    typedef pcl::PointXYZRGB PointT;
    
    PointCloudFilterUtils();
    virtual void cloudCallback(
        const sensor_msgs::PointCloud2::ConstPtr &);

    virtual void onInit();
    virtual void subsribe();
    virtual void pclDistanceFilter(
        const boost::shared_ptr<pcl::PCLPointCloud2>,
        pcl::PCLPointCloud2 &);
    virtual void configCallback(
       pcl_filter_utils::PointCloudFilterUtilsConfig &, uint32_t);
    
 protected:
    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_cloud_;
    ros::Publisher pub_cloud_;

    dynamic_reconfigure::Server<
       pcl_filter_utils::PointCloudFilterUtilsConfig>  server;
    
 private:
    float min_distance_;
    float max_distance_;
};

#endif  // _PCL_FILTER_UTILS_H_
