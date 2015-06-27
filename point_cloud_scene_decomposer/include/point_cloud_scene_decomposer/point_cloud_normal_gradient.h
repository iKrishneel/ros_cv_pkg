
#ifndef _POINT_CLOUD_NORMAL_GRADIENT_H_
#define _POINT_CLOUD_NORMAL_GRADIENT_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <sensor_msgs/PointCloud2.h>
#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d_omp.h>

class PointCloudNormalGradients{
   
 public:
    typedef pcl::PointXYZRGB PointT;
    PointCloudNormalGradients();
    void cloudCallback(
       const sensor_msgs::PointCloud2::ConstPtr &);
    void viewPointSurfaceNormalOrientation(
       pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::Normal>::Ptr);
    void estimatePointCloudNormals(
       const pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<pcl::Normal>::Ptr,
       const int = 8,
       const double = 0.03,
       bool = true);

    void localCurvatureBoundary(
      pcl::PointCloud<PointT>::Ptr,
      const pcl::PointCloud<pcl::Normal>::Ptr);
    void pclNearestNeigborSearch(
       pcl::PointCloud<PointT>::Ptr, std::vector<std::vector<int> > &,
       bool isneigbour = true, const int = 8, const double = 0.05);
    void convertToRvizNormalDisplay(
      const pcl::PointCloud<PointT>::Ptr,
      const pcl::PointCloud<pcl::Normal>::Ptr,
      pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr);
   
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
   
 private:
    ros::NodeHandle nh_;
    ros::Publisher pub_norm_;
    ros::Publisher pub_cloud_;
    ros::Publisher pub_norm_xyz_;
    ros::Subscriber sub_cloud_;
   
    template<typename T, typename U, typename V>
    cv::Scalar JetColour(T, U, V);
};


#endif  // _POINT_CLOUD_NORMAL_GRADIENT_H_
