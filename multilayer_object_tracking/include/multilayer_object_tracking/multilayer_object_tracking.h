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
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <boost/thread/mutex.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>
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
#include <pcl/features/gfpfh.h>
#include <pcl/features/pfh.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/tracking/tracking.h>
#include <pcl/common/common.h>
#include <pcl/registration/distances.h>
#include <pcl/filters/passthrough.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/PointsArray.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Header.h>

#include <multilayer_object_tracking/supervoxel_segmentation.h>
#include <multilayer_object_tracking/ReferenceModelBundle.h>
#include <multilayer_object_tracking/EstimatedCentroidsClustering.h>
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
        Eigen::Vector3f centroid_distance;  // not used in ref model
       cv::Mat neigbour_pfh;
       int query_index;  // used for holding test-target match index
       bool flag;
        uint32_t supervoxel_index;
        std::vector<int> history_window;
        int match_counter;
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
        sensor_msgs::PointCloud2,
        geometry_msgs::PoseStamped> ObjectSyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_obj_cloud_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_bkgd_cloud_;
    message_filters::Subscriber<geometry_msgs::PoseStamped> sub_obj_pose_;
    boost::shared_ptr<
       message_filters::Synchronizer<ObjectSyncPolicy> >obj_sync_;
   
    ros::Publisher pub_cloud_;
    ros::Publisher pub_templ_;
    ros::Publisher pub_sindices_;
    ros::Publisher pub_scloud_;
    ros::Publisher pub_normal_;
    ros::Publisher pub_tdp_;
    ros::Publisher pub_inliers_;
    ros::Publisher pub_centroids_;
    ros::Publisher pub_pose_;
    ros::Publisher pub_prob_;
    ros::Publisher pub_template_set_;
    ros::ServiceClient clustering_client_;
    
    // object model params
    int init_counter_;
    ModelsPtr object_reference_;
    ModelsPtr background_reference_;
    
    // motion previous
    MotionHistory motion_history_;
    int update_counter_;
    
    // hold current position
    Eigen::Vector4f current_pose_;
    Eigen::Vector4f previous_pose_;
    PointXYZRPY tracker_pose_;  // temp variable remove later

    // save previous tf
    tf::Transform previous_transform_;

    pcl::PointCloud<PointT>::Ptr previous_template_;
    float growth_rate_;
    float previous_distance_;
    
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
       const sensor_msgs::PointCloud2::ConstPtr &,
       const geometry_msgs::PoseStamped::ConstPtr &);
   
    virtual std::vector<pcl::PointIndices::Ptr>
    clusterPointIndicesToPointIndices(
       const jsk_recognition_msgs::ClusterPointIndicesConstPtr &);
    void estimatedPFPose(
       const geometry_msgs::PoseStamped::ConstPtr &, PointXYZRPY &);

    void voxelizeAndProcessPointCloud(
       const pcl::PointCloud<PointT>::Ptr cloud,
       const std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr> &,
       const std::multimap<uint32_t, uint32_t> &,
       std::vector<AdjacentInfo> &,
       ModelsPtr &, bool = true, bool = true, bool = true, bool = false);

    void targetDescriptiveSurfelsEstimationAndUpdate(
       pcl::PointCloud<PointT>::Ptr,
       const Eigen::Affine3f &,
       const MultilayerObjectTracking::PointXYZRPY &,
       const std_msgs::Header);
    template<class T>
    T targetCandidateToReferenceLikelihood(
       const ReferenceModel &,
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::Normal>::Ptr,
       const Eigen::Vector4f &,
       ReferenceModel * = NULL);
   
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
       const int = 16,
       const int = 16,
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
    void estimatedCentroidClustering(
       const std::multimap<uint32_t, Eigen::Vector3f> &,
       pcl::PointCloud<PointT>::Ptr,
       std::vector<uint32_t> &,
       std::vector<uint32_t> &);
    
    void computeLocalPairwiseFeautures(
        const std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr> &,
        const std::map<uint32_t, std::vector<uint32_t> >&,
        cv::Mat &, const int = 3);
    void processVoxelForReferenceModel(
        const std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr>,
        const std::multimap<uint32_t, uint32_t>,
        const uint32_t, MultilayerObjectTracking::ReferenceModel *);
    void transformModelPrimitives(
        const ModelsPtr &,
        ModelsPtr,
        const Eigen::Affine3f &);
    float templateCloudFilterLenght(
        const pcl::PointCloud<PointT>::Ptr);
    bool filterPointCloud(
        pcl::PointCloud<PointT>::Ptr,
        const Eigen::Vector4f,
        const ModelsPtr,
        const float);
    void processInitCloud(
        const pcl::PointCloud<PointT>::Ptr,
        ModelsPtr);
    void backgroundReferenceLikelihood(
        const ModelsPtr,
        const ModelsPtr,
        std::map<uint32_t, float>);
    void filterCloudForBoundingBoxViz(
        pcl::PointCloud<PointT>::Ptr,
        const ModelsPtr,
        const float = 0.6f);
    
    void computeScatterMatrix(
       const pcl::PointCloud<PointT>::Ptr,
       const Eigen::Vector4f);

    template<typename T, typename U, typename V>
    cv::Scalar plotJetColour(T, U, V);

    jsk_recognition_msgs::PointsArray
    convertAndPublishTemplateAsJSKPointsArray(
       const ModelsPtr, const std_msgs::Header);
};

#endif  // _MULTILAYER_OBJECT_TRACKING_H_
