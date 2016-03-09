
#ifndef _POINT_CLOUD_EDGE_H_
#define _POINT_CLOUD_EDGE_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <boost/thread/mutex.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/thread/thread.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/features/integral_image_normal.h>

#include <pcl_msgs/PointIndices.h>

// #if PCL_VERSION_COMPARE(<, 1, 7, 2)
#include <point_cloud_edge/organized_edge_detection.h>
// #endif
// #if PCL_VERSION_COMPARE(=, 1, 7, 2)
// #include <pcl/features/organized_edge_detection.h>
// #endif
class PointCloudEdge {
   
 private:
    typedef pcl::PointXYZRGB PointT;
 protected:
    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Publisher pub_hc_edge_;
    ros::Publisher pub_oc_edge_;
    ros::Subscriber sub_cloud_;

    ros::Publisher pub_occluding_indices_;
    ros::Publisher pub_curvature_indices_;
   
 public:
    PointCloudEdge();
    virtual void onInit();
    virtual void subscribe();
    virtual void callback(
       const sensor_msgs::PointCloud2::ConstPtr &);
    void publishIndices(
       ros::Publisher&, ros::Publisher&,
       const pcl::PointCloud<PointT>::Ptr&,
       const std::vector<int>&, const std_msgs::Header&);
   
};
#endif  // _POINT_CLOUD_EDGE_H_
