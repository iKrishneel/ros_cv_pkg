
#ifndef _POINT_CLOUD_MINCUT_MAXFLOW_
#define _POINT_CLOUD_MINCUT_MAXFLOW_

#include <ros/ros.h>
#include <ros/console.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>

#include <omp.h>

class PointCloudMinCutMaxFlow {
   
 private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
    typedef message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2,
       sensor_msgs::PointCloud2> SyncPolicy;
   
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_mask_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    int num_threads_;
   
 protected:
    ros::Publisher pub_cloud_;
    ros::Publisher pub_obj_;
    ros::Publisher pub_indices_;

    void onInit();
    void subscribe();
    void unsubscribe();
   
 public:
    typedef pcl::PointXYZRGB PointT;
    PointCloudMinCutMaxFlow();
    void callback(
       const sensor_msgs::PointCloud2::ConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &);
};


#endif  // _POINT_CLOUD_MINCUT_MAXFLOW_
