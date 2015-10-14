
#ifndef _FEATURE_BASED_REGISTERATION_H_
#define _FEATURE_BASED_REGISTERATION_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <boost/make_shared.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/point_representation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/keypoints/sift_keypoint.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/gicp.h>

class FeatureBasedRegisteration {

    typedef pcl::PointXYZRGB PointT;

 public:
    FeatureBasedRegisteration();
    void callback(
       const sensor_msgs::PointCloud2::ConstPtr &);
    void imageCallback(
       const sensor_msgs::Image::ConstPtr &);
   
    template<class T>
    void estimatePointCloudNormals(
       const pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<pcl::PointNormal>::Ptr,
       const T, bool = false) const;
    void computePointFPFH(
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::PointNormal>::Ptr,
       cv::Mat &, bool) const;

    void keypointsFrom2DImage(
       const pcl::PointCloud<PointT>::Ptr,
       const cv::Mat &);
    void getPointCloudKeypoints(
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::PointNormal>::Ptr,
       pcl::PointCloud<pcl::PointWithScale>::Ptr result,
       const float = 0.01f,
       const int = 3,
       const int = 4,
       const float = 0.001f);
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
   
 private:
    ros::NodeHandle nh_;
    ros::Publisher pub_cloud_;
    ros::Publisher pub_regis_;
    ros::Subscriber sub_cloud_;
    ros::Subscriber sub_image_;
    pcl::PointCloud<PointT>::Ptr reg_cloud;
    cv::Mat image;
   
};

#endif  // _FEATURE_BASED_REGISTERATION_H_

