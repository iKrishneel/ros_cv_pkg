// Copyright (C) 2015 by Krishneel Chaudhary

#ifndef _POINT_CLOUD_SCENE_DECOMPOSER_H_ 
#define _POINT_CLOUD_SCENE_DECOMPOSER_H_

#include <point_cloud_scene_decomposer/constants.h>
#include <point_cloud_scene_decomposer/scene_decomposer_image_processor.h>
#include <point_cloud_scene_decomposer/region_adjacency_graph.h>
#include <point_cloud_scene_decomposer/signal.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <geometry_msgs/PoseArray.h>

class PointCloudSceneDecomposer: public SceneDecomposerImageProcessor {
 private:
    ros::NodeHandle nh_;
    ros::Publisher pub_image_;
    ros::Publisher pub_cloud_;
    ros::Publisher pub_indices_;
    ros::Publisher pub_cloud_orig_;
   
    ros::Subscriber sub_cloud_;
    ros::Subscriber sub_cloud_ori_;
    ros::Subscriber sub_norm_;
    ros::Subscriber sub_image_;
    ros::Subscriber sub_indices_;
    // ros::Subscriber sub_bbox_;
   
    ros::Subscriber sub_signal_;
    ros::Publisher pub_signal_;
    // ros::Publisher pub_known_bbox_;
   
    pcl::PointCloud<PointT>::Ptr pcl_cloud__;
    pcl::PointCloud<PointT>::Ptr filter_cloud__;

    bool start_signal_;
    int processing_counter_;
    point_cloud_scene_decomposer::signal signal_;
   
    //  variables to publish while not processing
    jsk_recognition_msgs::ClusterPointIndices publishing_indices;
    sensor_msgs::PointCloud2 publishing_cloud;
    sensor_msgs::PointCloud2 publishing_cloud_orig;
    cv_bridge::CvImagePtr image_msg;
   
    void pclNearestNeigborSearch(
        pcl::PointCloud<pcl::PointXYZ>::Ptr,
        std::vector<std::vector<int> > &,
        bool isneigbour = true,
        const int = 8,
        const double = 0.05);

    void semanticCloudLabel(
        const std::vector<pcl::PointCloud<PointT>::Ptr> &,
        pcl::PointCloud<PointT>::Ptr,
        const std::vector<int> &,
        std::vector<pcl::PointIndices> &,
        const int);

    std::vector<pcl_msgs::PointIndices> convertToROSPointIndices(
       const std::vector<pcl::PointIndices>,
       const std_msgs::Header&);
   
    float max_distance_;
   
 protected:

    void onInit();
    void subscribe();
    void unsubscribe();

    pcl::PointCloud<pcl::Normal>::Ptr normal_;
    pcl::PointCloud<PointT>::Ptr orig_cloud_;
    cv::Mat image_;
    
 public:
    PointCloudSceneDecomposer();
    void origcloudCallback(
       const sensor_msgs::PointCloud2ConstPtr &);
    void cloudCallback(
        const sensor_msgs::PointCloud2ConstPtr &);
    void normalCallback(
       const sensor_msgs::PointCloud2ConstPtr &);
    void signalCallback(
       const point_cloud_scene_decomposer::signal &);
    void imageCallback(
       const sensor_msgs::Image::ConstPtr &);
    void indicesCallback(
       const jsk_recognition_msgs::ClusterPointIndices &);   
   
    void extractPointCloudClustersFrom2DMap(
        const pcl::PointCloud<PointT>::Ptr,
        const std::vector<cvPatch<int> > &,
        std::vector<pcl::PointCloud<PointT>::Ptr> &,
        std::vector<pcl::PointCloud<pcl::Normal>::Ptr> &,
        pcl::PointCloud<pcl::PointXYZ>::Ptr,
        const cv::Size);


    void pointCloudVoxelClustering(
       std::vector<pcl::PointCloud<PointT>::Ptr> &,
       const std::vector<pcl::PointCloud<pcl::Normal>::Ptr> &,
       const pcl::PointCloud<pcl::PointXYZ>::Ptr,
       std::vector<int> &);
    void clusterVoxels(
       const cv::Mat &,
       std::vector<int> &);

    virtual void objectCloudClusterPostProcessing(
       pcl::PointCloud<PointT>::Ptr,
       std::vector<pcl::PointIndices> &,
       const int = 64);
};
#endif  // _POINT_CLOUD_SCENE_DECOMPOSER_H_
