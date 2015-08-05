// Copyright (C) 2015 by Krishneel Chaudhary, JSK Lab,
// The University of Tokyo, Japan

#ifndef _MULTILAYER_OBJECT_TRACKING_H_
#define _MULTILAYER_OBJECT_TRACKING_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/video/background_segm.hpp>

#include <boost/thread/mutex.hpp>

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
#include <pcl/segmentation/segment_differences.h>
#include <pcl/octree/octree.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/filter.h>
#include <pcl/common/centroid.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/vfh.h>
#include <pcl/tracking/tracking.h>
#include <pcl/common/common.h>
#include <pcl/registration/distances.h>
#include <pcl/features/gfpfh.h>

#include <tf/transform_listener.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Header.h>

#include <multilayer_object_tracking/supervoxel_segmentation.h>
#include <map>

class MultilayerObjectTracking: public SupervoxelSegmentation {
    typedef pcl::PointXYZRGB PointT;

    struct AdjacentInfo {
       /* -old type (remove) */
       std::vector<int> adjacent_indices;
       std::vector<float> adjacent_distances;
       
       uint32_t voxel_index;
       std::map<uint32_t, std::vector<uint32_t> > adjacent_voxel_indices;
    };
   
    struct ReferenceModel {
       pcl::PointCloud<PointT>::Ptr cluster_cloud;
       cv::Mat cluster_vfh_hist;
       cv::Mat cluster_color_hist;
       AdjacentInfo cluster_neigbors;
       pcl::PointCloud<pcl::Normal>::Ptr cluster_normals;
       Eigen::Vector4f cluster_centroid;
       Eigen::Vector3f centroid_distance;
       bool flag;
    };
    typedef std::vector<ReferenceModel> Models;
    typedef boost::shared_ptr<Models> ModelsPtr;
   
    typedef pcl::tracking::ParticleXYZRPY PointXYZRPY;
    typedef std::vector<PointXYZRPY> MotionHistory;
   
 private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2,
       geometry_msgs::PoseStamped> SyncPolicy;
   
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<geometry_msgs::PoseStamped> sub_pose_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    // subscribe after init
    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2,
       geometry_msgs::PoseStamped> ObjectSyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_obj_cloud_;
    message_filters::Subscriber<geometry_msgs::PoseStamped> sub_obj_pose_;
    boost::shared_ptr<
       message_filters::Synchronizer<ObjectSyncPolicy> >obj_sync_;
   
    ros::Publisher pub_cloud_;
    ros::Publisher pub_sindices_;
    ros::Publisher pub_scloud_;
    ros::Publisher pub_normal_;
    ros::Publisher pub_tdp_;
    
    // object model params
    int init_counter_;
    int min_cluster_size_;
    ModelsPtr object_reference_;
   
    float threshold_;

    // motion previous
    MotionHistory motion_history_;
    
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();

 public:
    MultilayerObjectTracking();
    virtual void callback(
       const sensor_msgs::PointCloud2::ConstPtr &,
       const geometry_msgs::PoseStamped::ConstPtr &);
    virtual void objInitCallback(
       const sensor_msgs::PointCloud2::ConstPtr &,
       const geometry_msgs::PoseStamped::ConstPtr &);
   
    virtual std::vector<pcl::PointIndices::Ptr>
    clusterPointIndicesToPointIndices(
       const jsk_recognition_msgs::ClusterPointIndicesConstPtr &);
    void estimatedPFPose(
       const geometry_msgs::PoseStamped::ConstPtr &, PointXYZRPY &);

    void processDecomposedCloud(
       const pcl::PointCloud<PointT>::Ptr cloud,
       const std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr> &,
       const std::multimap<uint32_t, uint32_t> &,
       std::vector<AdjacentInfo> &,
       ModelsPtr &, bool = true, bool = true, bool = true);

   /*
    std::vector<AdjacentInfo> voxelAdjacencyList(
       const jsk_recognition_msgs::AdjacencyList &);
   */
    void globalLayerPointCloudProcessing(
       pcl::PointCloud<PointT>::Ptr,
       const MultilayerObjectTracking::PointXYZRPY &,
       const std_msgs::Header);
    template<class T>
    T targetCandidateToReferenceLikelihood(
       const ReferenceModel &,
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::Normal>::Ptr,
       const Eigen::Vector4f &);
    template<class T>
    T localVoxelConvexityLikelihood(
       Eigen::Vector4f,
       Eigen::Vector4f,
       Eigen::Vector4f,
       Eigen::Vector4f);
   
    template<class T>
    void estimatePointCloudNormals(
       const pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<pcl::Normal>::Ptr,
       T = 0.05f, bool = false) const;
   
    void computeCloudClusterRPYHistogram(
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::Normal>::Ptr,
       cv::Mat &) const;
    void computeColorHistogram(
       const pcl::PointCloud<PointT>::Ptr,
       cv::Mat &,
       const int = 8,
       const int = 8,
       bool = true) const;
    void computePointFPFH(
       const pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<pcl::Normal>::Ptr,
       cv::Mat &, bool = true) const;
    void compute3DCentroids(
       const pcl::PointCloud<PointT>::Ptr,
       Eigen::Vector4f &) const;
    Eigen::Vector4f cloudMeanNormal(
       const pcl::PointCloud<pcl::Normal>::Ptr, bool = true);
    void adjacentVoxelCoherencey(
       const Models &, const int, float &, float &);
    float computeCoherency(
       const float, const float);
    pcl::PointXYZRGBNormal
    convertVector4fToPointXyzRgbNormal(
        const Eigen::Vector4f &,
        const Eigen::Vector4f &,
        const cv::Scalar);

    template<typename T>
    void getRotationMatrixFromRPY(
        const PointXYZRPY &,
        Eigen::Matrix<T, 3, 3> &);
        
    void computeScatterMatrix(
       const pcl::PointCloud<PointT>::Ptr,
       const Eigen::Vector4f);
   
};

#endif  // _MULTILAYER_OBJECT_TRACKING_H_
