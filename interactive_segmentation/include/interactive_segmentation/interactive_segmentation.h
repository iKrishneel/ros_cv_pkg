// Copyright (C) 2015 by Krishneel Chaudhary, JSK Lab,
// The University of Tokyo, Japan

#ifndef _INTERACTIVE_SEGMENTATION_H_
#define _INTERACTIVE_SEGMENTATION_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <boost/thread/mutex.hpp>
#include <boost/graph/grid_graph.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/iteration_macros.hpp>

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
#include <jsk_recognition_msgs/PolygonArray.h>
#include <jsk_recognition_msgs/Int32Stamped.h>

#include <std_msgs/Header.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int64.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PointStamped.h>

#include <jsk_recognition_utils/pcl_conversion_util.h>
#include <jsk_recognition_utils/geo/polygon.h>
#include <jsk_perception/skeletonization.h>

#include <omp.h>

class InteractiveSegmentation {

    typedef pcl::PointXYZRGB PointT;
    typedef  pcl::FPFHSignature33 FPFHS;

 private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
    typedef message_filters::sync_policies::ApproximateTime<
       sensor_msgs::Image,
       sensor_msgs::CameraInfo,
       sensor_msgs::PointCloud2,
       sensor_msgs::PointCloud2> SyncPolicy;

    message_filters::Subscriber<sensor_msgs::Image> sub_image_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_normal_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> sub_info_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    // user mark the point
    typedef message_filters::sync_policies::ApproximateTime<
      geometry_msgs::PointStamped,
      sensor_msgs::PointCloud2> UsrSyncPolicy;

    message_filters::Subscriber<geometry_msgs::PointStamped> sub_screen_pt_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_orig_cloud_;
    boost::shared_ptr<message_filters::Synchronizer<UsrSyncPolicy> >usr_sync_;

    ros::Subscriber sub_polyarray_;
   
    ros::Publisher pub_cloud_;
    ros::Publisher pub_convex_;
    ros::Publisher pub_concave_;
    ros::Publisher pub_normal_;
    ros::Publisher pub_prob_;
    ros::Publisher pub_apoints_;
    ros::Publisher pub_indices_;
    ros::Publisher pub_image_;
    ros::Publisher pub_pt_map_;
    ros::Publisher pub_plane_;
   
    ros::ServiceClient srv_client_;
    PointT user_marked_pt_;
    cv::Point2i screen_pt_;
    bool is_init_;
    bool is_stop_signal_;
  
    int min_cluster_size_;
    int num_threads_;
  
    sensor_msgs::CameraInfo::ConstPtr camera_info_;
    jsk_recognition_msgs::PolygonArray polygon_array_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
   
 public:
    InteractiveSegmentation();
    virtual void screenPointCallback(
       const geometry_msgs::PointStamped::ConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &);
    virtual void polygonArrayCallback(
       const jsk_recognition_msgs::PolygonArray::ConstPtr &);
    virtual void callback(
       const sensor_msgs::Image::ConstPtr &,
       const sensor_msgs::CameraInfo::ConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &);
    void selectedVoxelObjectHypothesis(
       pcl::PointCloud<PointT>::Ptr, const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::Normal>::Ptr,
       const pcl::PointIndices::Ptr, const std_msgs::Header);
    bool attentionSurfelRegionPointCloudMask(
       const pcl::PointCloud<PointT>::Ptr, const Eigen::Vector4f,
       pcl::PointCloud<PointT>::Ptr, pcl::PointIndices::Ptr);
    void surfelSamplePointWeightMap(
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::Normal>::Ptr, const PointT &,
       const Eigen::Vector4f, cv::Mat &);
    void computePointCloudCovarianceMatrix(
       const pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<PointT>::Ptr);
    float whiteNoiseKernel(
       const float, const float = 0.0f, const float = 0.0f);
    virtual Eigen::Vector4f cloudMeanNormal(
       const pcl::PointCloud<pcl::Normal>::Ptr, bool = false);
    int localVoxelConvexityCriteria(
       Eigen::Vector4f, Eigen::Vector4f, Eigen::Vector4f, const float = 0.0f);
    template<class T>
    void estimatePointCloudNormals(
       const pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<pcl::Normal>::Ptr,
       T = 0.05f, bool = false) const;
    void pointIntensitySimilarity(
       pcl::PointCloud<PointT>::Ptr, const int);
    void selectedPointToRegionDistanceWeight(
       const pcl::PointCloud<PointT>::Ptr, const Eigen::Vector3f,
       const float, const sensor_msgs::CameraInfo::ConstPtr);
    cv::Mat projectPointCloudToImagePlane(
       const pcl::PointCloud<PointT>::Ptr,
       const sensor_msgs::CameraInfo::ConstPtr &, cv::Mat &, cv::Mat &);
    void highCurvatureEdgeBoundary(
       pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::Normal>::Ptr, const std_msgs::Header);
    bool estimateAnchorPoints(
       pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<PointT>::Ptr, pcl::PointIndices::Ptr,
       pcl::PointIndices::Ptr, Eigen::Vector4f &,
       const pcl::PointCloud<PointT>::Ptr);
    std::vector<Eigen::Vector4f> doEuclideanClustering(
       std::vector<pcl::PointIndices> &cluster_indices,
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointIndices::Ptr, bool = false, const float = 0.02f,
       const int = 50, const int = 20000);
    void edgeBoundaryOutlierFiltering(
        const pcl::PointCloud<PointT>::Ptr,
        const float = 0.01f, const int = 50);
    bool skeletonization2D(
        pcl::PointCloud<PointT>::Ptr, pcl::PointIndices::Ptr,
       const pcl::PointCloud<PointT>::Ptr,
       const sensor_msgs::CameraInfo::ConstPtr &, const cv::Scalar);
    std::vector<Eigen::Vector4f> thinBoundaryAndComputeCentroid(
        pcl::PointCloud<PointT>::Ptr,
        const pcl::PointCloud<PointT>::Ptr,
        std::vector<pcl::PointIndices> &, const cv::Scalar);
    void publishAsROSMsg(
        const pcl::PointCloud<PointT>::Ptr, const ros::Publisher,
        const std_msgs::Header);
    void filterAndComputeNonObjectRegionAnchorPoint(
        pcl::PointCloud<PointT>::Ptr,
        const pcl::PointCloud<pcl::Normal>::Ptr,
        const int, const cv::Mat &, const float = 0.0f);
    void fixPlaneModelToEdgeBoundaryPoints(
       pcl::PointCloud<PointT>::Ptr, pcl::PointIndices::Ptr,
       Eigen::Vector3f &, const Eigen::Vector4f);
    void supportPlaneNormal(const Eigen::Vector4f, const std_msgs::Header);
    bool markedPointInSegmentedRegion(
       const pcl::PointCloud<PointT>::Ptr, const PointT);
   
    void normalizedCurvatureNormalHistogram(
       pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<pcl::Normal>::Ptr);
};


#endif  // _INTERACTIVE_SEGMENTATION_H_
