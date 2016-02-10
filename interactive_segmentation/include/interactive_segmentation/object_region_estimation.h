// Copyright (C) 2016 by Krishneel Chaudhary, JSK Lab,a
// The University of Tokyo, Japan

#ifndef _OBJECT_REGION_ESTIMATION_H_
#define _OBJECT_REGION_ESTIMATION_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

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
#include <pcl/filters/filter.h>
#include <pcl/common/centroid.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/point_types_conversion.h>
#include <pcl/registration/distances.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/features/shot.h>
#include <pcl/features/shot_omp.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/board.h>
#include <pcl/recognition/cg/hough_3d.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_utils/geo/polygon.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <jsk_recognition_msgs/Histogram.h>

#include <std_msgs/Header.h>
#include <geometry_msgs/PoseStamped.h>

#include <interactive_segmentation/Feature3DClustering.h>

#include <omp.h>

class ObjectRegionEstimation {

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointXYZI PointI;
    typedef pcl::Normal Normal;
    typedef pcl::SHOT352 SHOT352;
    typedef pcl::SHOT1344 SHOT1344;
    typedef pcl::ReferenceFrame RFType;
   
#define FEATURE_DIM 352
  
 private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
  
    typedef message_filters::sync_policies::ApproximateTime<
       sensor_msgs::Image,
       sensor_msgs::PointCloud2,
       sensor_msgs::PointCloud2,
       geometry_msgs::PoseStamped> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::Image> sub_image_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_original_;
    message_filters::Subscriber<geometry_msgs::PoseStamped> sub_pose_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    typedef message_filters::sync_policies::ApproximateTime<
       sensor_msgs::Image,
       sensor_msgs::PointCloud2,
       sensor_msgs::PointCloud2> SyncPolicyPrev;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_prev_;
    message_filters::Subscriber<sensor_msgs::Image> sub_image_prev_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_plane_prev_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicyPrev> >sync_prev_;
  
    ros::Publisher pub_cloud_;
    ros::Publisher pub_indices_;
    ros::ServiceClient srv_client_;
  
    int num_threads_;
    bool is_prev_ok;
    float min_cluster_size_;
    std_msgs::Header header_;
  
    pcl::PointCloud<PointT>::Ptr prev_cloud_;
    cv::Mat prev_image_;
    Eigen::Vector3f plane_norm_;
    Eigen::Vector3f plane_point_;

    bool go_signal_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
   
 public:
    ObjectRegionEstimation();
    virtual void callback(
       const sensor_msgs::Image::ConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &,
       const geometry_msgs::PoseStamped::ConstPtr &);
    virtual void callbackPrev(
       const sensor_msgs::Image::ConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &);
       
    virtual void sceneFlow(
       pcl::PointCloud<PointT>::Ptr, const cv::Mat, const cv::Mat,
       const pcl::PointCloud<PointT>::Ptr, const Eigen::Vector3f,
       const Eigen::Vector3f);
    template<class T>
    void estimatePointCloudNormals(
       const pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<Normal>::Ptr,
       T = 0.05f, bool = false) const;

   
    void keypoints3D(
        pcl::PointCloud<PointI>::Ptr, const pcl::PointCloud<PointT>::Ptr);
    void features3D(
       pcl::PointCloud<SHOT1344>::Ptr, const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<Normal>::Ptr, const pcl::PointCloud<PointI>::Ptr);
    void getHypothesis(
        const pcl::PointCloud<PointT>::Ptr, const pcl::PointCloud<PointT>::Ptr);

    void noiseClusterFilter(
       pcl::PointCloud<PointT>::Ptr, pcl::PointIndices::Ptr indices,
       const pcl::PointCloud<PointT>::Ptr);
   
    void removeStaticKeypoints(
       pcl::PointCloud<PointI>::Ptr, pcl::PointCloud<PointI>::Ptr,
       const float = 0.01f);
    void clusterFeatures(
        std::vector<pcl::PointIndices> &, const pcl::PointCloud<PointT>::Ptr,
        const pcl::PointCloud<Normal>::Ptr, const int, const float);

    void drawOptFlowMap(
        const cv::Mat&, cv::Mat&, int, double, const cv::Scalar&);
};


#endif   // _OBJECT_REGION_ESTIMATION_H_
