
#ifndef _SUPERVOXEL_SEGMENTATION_H_
#define _SUPERVOXEL_SEGMENTATION_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/supervoxel_clustering.h>

#include <dynamic_reconfigure/server.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Header.h>
#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <multilayer_object_tracking/SupervoxelSegmentationConfig.h>

class SupervoxelSegmentation {
 protected:
    double color_importance_;
    double spatial_importance_;
    double normal_importance_;
    double voxel_resolution_;
    double seed_resolution_;
    bool use_transform_;

    int min_cluster_size_;
    float threshold_;
    int bin_size_;
    float eps_distance_;
    int eps_min_samples_;
    float vfh_scaling_;
    float color_scaling_;
    float structure_scaling_;
    bool update_tracker_reference_;
    bool update_filter_template_;
    boost::mutex mutex_;
   
 public:
    SupervoxelSegmentation();
    typedef pcl::PointXYZRGB PointT;
    typedef boost::shared_ptr<SupervoxelSegmentation> Ptr;

    void supervoxelSegmentation(
       const pcl::PointCloud<PointT>::Ptr,
       std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > &,
       std::multimap<uint32_t, uint32_t> &);
    void publishSupervoxel(
       const std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>,
       sensor_msgs::PointCloud2 &,
       jsk_recognition_msgs::ClusterPointIndices &,
       const std_msgs::Header &);
    std::vector<pcl_msgs::PointIndices> convertToROSPointIndices(
       const std::vector<pcl::PointIndices>,
       const std_msgs::Header &);
    void targetDescriptiveSurfelsIndices(
           const jsk_recognition_msgs::ClusterPointIndices &,
           const std::vector<uint32_t> &,
           jsk_recognition_msgs::ClusterPointIndices &);   
   
 private:
    typedef multilayer_object_tracking::SupervoxelSegmentationConfig Config;
    virtual void configCallback(Config &, uint32_t);
    boost::shared_ptr<dynamic_reconfigure::Server<Config> > srv_;
};


#endif  //_SUPERVOXEL_SEGMENTATION_H_
