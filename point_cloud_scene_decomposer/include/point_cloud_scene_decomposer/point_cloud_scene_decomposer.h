// Copyright (C) 2015 by Krishneel Chaudhary

#ifndef _POINT_CLOUD_SCENE_DECOMPOSER_H_ 
#define _POINT_CLOUD_SCENE_DECOMPOSER_H_

#include <point_cloud_scene_decomposer/constants.h>
#include <point_cloud_scene_decomposer/scene_decomposer_image_processor.h>
#include <point_cloud_scene_decomposer/region_adjacency_graph.h>

class PointCloudSceneDecomposer: public SceneDecomposerImageProcessor {
 private:
    ros::NodeHandle nh_;
    ros::Publisher pub_image_;
    ros::Publisher pub_cloud_;
    ros::Subscriber sub_cloud_;
    ros::Subscriber sub_cloud_ori_;
    ros::Subscriber sub_norm_;
    ros::Subscriber sub_image_;

    pcl::PointCloud<PointT>::Ptr pcl_cloud__;
    pcl::PointCloud<PointT>::Ptr filter_cloud__;

    void pclNearestNeigborSearch(
        pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<std::vector<int> > &,
        bool isneigbour = true, const int = 8, const double = 0.05);


    void semanticCloudLabel(
        const std::vector<pcl::PointCloud<PointT>::Ptr> &,
        pcl::PointCloud<PointT>::Ptr, const std::vector<int> &);
   
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
    void pointCloudLocalGradient(
        const pcl::PointCloud<PointT>::Ptr,
        const pcl::PointCloud<pcl::Normal>::Ptr,
        cv::Mat &);
    void extractPointCloudClustersFrom2DMap(
        const pcl::PointCloud<PointT>::Ptr,
        const std::vector<cvPatch<int> > &,
        std::vector<pcl::PointCloud<PointT>::Ptr> &,
        std::vector<pcl::PointCloud<pcl::Normal>::Ptr> &,
        pcl::PointCloud<pcl::PointXYZ>::Ptr,
        const cv::Size);

    void imageCallback(
        const sensor_msgs::Image::ConstPtr &);
    
};
#endif  // _POINT_CLOUD_SCENE_DECOMPOSER_H_
