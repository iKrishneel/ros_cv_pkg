
#ifndef _INCREMENTAL_ICP_REGISTERATION_H_
#define _INCREMENTAL_ICP_REGISTERATION_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <sensor_msgs/PointCloud2.h>
#include <opencv2/opencv.hpp>
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
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>


class ICPPointRepresentation: public pcl::PointRepresentation<
    pcl::PointNormal> {
    using pcl::PointRepresentation<pcl::PointNormal>::nr_dimensions_;
 public:
    ICPPointRepresentation() {
      nr_dimensions_ = 4;
    }
    virtual void copyToFloatArray(
      const pcl::PointNormal &p, float * out) const {
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
};


class IncrementalICPRegisteration {

    typedef pcl::PointXYZRGB PointT;
   
 public:
    IncrementalICPRegisteration();
    void callback(
       const sensor_msgs::PointCloud2::ConstPtr &);
    bool icpAlignPointCloud(
      const pcl::PointCloud<PointT>::Ptr,
      const pcl::PointCloud<PointT>::Ptr,
      pcl::PointCloud<PointT>::Ptr, Eigen::Matrix4f &);
    void downsampleCloud(
       const pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<PointT>::Ptr, const float = 0.05f);
    void estimateNormal(
       const pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<pcl::PointNormal>::Ptr,
       const int = 30);
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
   
 private:
    ros::NodeHandle nh_;
    ros::Publisher pub_cloud_;
    ros::Subscriber sub_cloud_;

    bool set_init;
    pcl::PointCloud<PointT>::Ptr initial_cloud;
};


#endif  // _INCREMENTAL_ICP_REGISTERATION_H_
