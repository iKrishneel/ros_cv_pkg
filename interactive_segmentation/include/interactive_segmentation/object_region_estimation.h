// Copyright (C) 2016 by Krishneel Chaudhary, JSK Lab,a
// The University of Tokyo, Japan

#ifndef _OBJECT_REGION_ESTIMATION_H_
#define _OBJECT_REGION_ESTIMATION_H_

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
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/octree/octree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/filter.h>
#include <pcl/common/centroid.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/surface/mls.h>
#include <pcl/point_types_conversion.h>
#include <pcl/registration/distances.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_utils/geo/polygon.h>
#include <jsk_recognition_msgs/PolygonArray.h>

class ObjectRegionEstimation {

    typedef pcl::PointXYZRGB PointT;

 private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
    typedef message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2,
       sensor_msgs::PointCloud2> SyncPolicy;

    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_indices_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
   
 public:
    ObjectRegionEstimation();
    virtual void callback(
       const sensor_msgs::PointCloud2::ConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &);
   
};


#endif   // _OBJECT_REGION_ESTIMATION_H_
