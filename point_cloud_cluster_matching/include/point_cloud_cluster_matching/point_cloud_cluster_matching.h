
// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University of Tokyo

#ifndef _POINT_CLOUD_CLUSTER_MATCHING_H_
#define _POINT_CLOUD_CLUSTER_MATCHING_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <opencv2/opencv.hpp>
#include <boost/thread/mutex.hpp>

#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>

#include <point_cloud_cluster_matching/point_cloud_image_creator.h>
#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int16.h>
#include <geometry_msgs/Pose.h>

#include <vector>

class PointCloudClusterMatching: public PointCloudImageCreator {

 private:
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::SHOT352 DescriptorType;
    typedef std::vector<Eigen::Matrix4f,
                        Eigen::aligned_allocator<
                           Eigen::Matrix4f> > AffineTrans;
   
    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_cloud_;
    ros::Subscriber sub_cloud_prev_;
    ros::Subscriber sub_indices_;
    ros::Subscriber sub_signal_;
    ros::Subscriber sub_cam_info_;
    ros::Subscriber sub_manip_cluster_;
    ros::Subscriber sub_grip_end_pose_;
    ros::Publisher pub_signal_;
    ros::Publisher pub_cloud_;

    int manipulated_cluster_index_;
    geometry_msgs::Pose gripper_pose_;
    std::vector<pcl::PointIndices> all_indices;
    std::vector<pcl::PointCloud<PointT>::Ptr> prev_cloud_clusters;
    sensor_msgs::CameraInfo::ConstPtr camera_info_;
   
 public:
    PointCloudClusterMatching();
    virtual void cloudCallback(
       const sensor_msgs::PointCloud2ConstPtr &);
    virtual void cloudPrevCallback(
      const sensor_msgs::PointCloud2ConstPtr &);
    virtual void indicesCallback(
      const jsk_recognition_msgs::ClusterPointIndices &);
    virtual void signalCallback(
      const std_msgs::Bool &);
    virtual void manipulatedClusterCallback(
      const std_msgs::Int16 &);
    virtual void gripperEndPoseCallback(
       const geometry_msgs::Pose &);
    virtual void cameraInfoCallback(
       const sensor_msgs::CameraInfo::ConstPtr &);

    virtual void createImageFromObjectClusters(
       const std::vector<pcl::PointCloud<PointT>::Ptr> &,
       const sensor_msgs::CameraInfo::ConstPtr,
       std::vector<cv::Mat> &);
   
   
    virtual void extractFeaturesAndMatchCloudPoints(
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<PointT>::Ptr,
       AffineTrans &,
       std::vector<pcl::Correspondences> &);
    virtual void objectCloudClusters(
       const pcl::PointCloud<PointT>::Ptr,
       const std::vector<pcl::PointIndices> &,
       std::vector<pcl::PointCloud<PointT>::Ptr> &);
   
    virtual void pointCloudNormal(
       const pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<pcl::Normal>::Ptr);
    virtual void getCloudClusterKeyPoints(
       const pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<PointT>::Ptr,
       const float);
    virtual void computeDescriptors(
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::Normal>::Ptr,
       pcl::PointCloud<DescriptorType>::Ptr,
       const float);
    virtual void modelSceneCorrespondences(
       pcl::PointCloud<DescriptorType>::Ptr,
       pcl::PointCloud<DescriptorType>::Ptr,
       pcl::CorrespondencesPtr,
       const float);
    virtual void HoughCorrespondanceClustering(
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::Normal>::Ptr,
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::Normal>::Ptr,
       const pcl::PointCloud<PointT>::Ptr,
       pcl::CorrespondencesPtr,
       AffineTrans &,
       std::vector<pcl::Correspondences> &,
       const float,
       const float,
       const float);
   
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
};

#endif  //  _POINT_CLOUD_CLUSTER_MATCHING_H_
