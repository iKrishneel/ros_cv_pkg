
#pragma once
#ifndef _PARTICLE_FILTER_TRACKING_H_
#define _PARTICLE_FILTER_TRACKING_H_

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/pcl_search.h>
#include <pcl/common/transforms.h>

#include <pcl/tracking/tracking.h>
#include <pcl/tracking/particle_filter.h>
#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>
#include <pcl/tracking/particle_filter_omp.h>
#include <pcl/tracking/coherence.h>
#include <pcl/tracking/distance_coherence.h>
#include <pcl/tracking/hsv_color_coherence.h>
#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/nearest_pair_point_cloud_coherence.h>

#include <boost/format.hpp>

class ParticleFilters {
   
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::tracking::ParticleXYZRPY ParticleT;
    typedef pcl::PointCloud<PointT> Cloud;
    typedef Cloud::Ptr CloudPtr;
    typedef Cloud::ConstPtr CloudConstPtr;
    typedef pcl::tracking::ParticleFilterTracker<
       PointT, ParticleT> ParticleFilter;

 private:
    typedef  message_filters::sync_policies::ApproximateTime<
    sensor_msgs::PointCloud2,
    // geometry_msgs::PoseStamped
    sensor_msgs::PointCloud2
    > SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_templ_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_pose_;
    // message_filters::Subscriber<geometry_msgs::PoseStamped> sub_pose_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    boost::mutex mtx_;
    boost::shared_ptr<ParticleFilter> tracker_;
    bool new_cloud_;
    double downsampling_grid_size_;
    int counter;
    bool tracker_init_;
   
 protected:
    ros::NodeHandle pnh_;
    void onInit();
    void subscribe();
    void unsubscribe();
   
    ros::Publisher pub_cloud_;
    ros::Publisher pub_pose_;

    ros::Subscriber sub_cloud_;
   
 public:
    ParticleFilters();
    void cloudCB(const sensor_msgs::PointCloud2::ConstPtr &);
    void templateCB(const sensor_msgs::PointCloud2::ConstPtr &,
                    // const geometry_msgs::PoseStamped::ConstPtr &
                    const sensor_msgs::PointCloud2::ConstPtr &);
   
    void filterPassThrough(const CloudConstPtr &, Cloud &);
    void gridSampleApprox(const CloudConstPtr &, Cloud &, double);
};



#endif /* _PARTICLE_FILTER_TRACKING_H_ */
