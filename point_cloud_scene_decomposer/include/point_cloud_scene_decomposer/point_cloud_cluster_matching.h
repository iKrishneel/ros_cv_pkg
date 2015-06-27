
// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University of Tokyo

#ifndef _POINT_CLOUD_CLUSTER_MATCHING_H_
#define _POINT_CLOUD_CLUSTER_MATCHING_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <image_geometry/pinhole_camera_model.h>
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

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <point_cloud_scene_decomposer/signal.h>

#include <std_msgs/Bool.h>
#include <std_msgs/Int64.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>

#include <vector>

class PointCloudClusterMatching {

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
   
    ros::Subscriber sub_image_;
    ros::Subscriber sub_image_prev_;
    ros::Subscriber sub_mask_;
    ros::Subscriber sub_mask_prev_;
   
    ros::Subscriber sub_manip_cluster_;
    ros::Subscriber sub_grip_end_pose_;
    ros::Subscriber sub_bbox_;
   
    ros::Publisher pub_signal_;
    ros::Publisher pub_cloud_;
    ros::Publisher pub_indices_;
    ros::Publisher pub_known_bbox_;
   
    cv::Mat image_;
    cv::Mat image_prev_;
    cv::Mat image_mask_;
    cv::Mat image_mask_prev_;
   
    int manipulated_cluster_index_;
    jsk_recognition_msgs::BoundingBoxArray bbox_;
    geometry_msgs::PoseStamped gripper_pose_;
    std::vector<pcl::PointIndices> all_indices_;
    std::vector<pcl::PointCloud<PointT>::Ptr> prev_cloud_clusters;
    pcl::PointCloud<PointT>::Ptr cloud_prev_;
   
    int depth_counter;
    int processing_counter_;

    sensor_msgs::PointCloud2 publishing_cloud;
    jsk_recognition_msgs::ClusterPointIndices publishing_indices;
    jsk_recognition_msgs::BoundingBoxArray publishing_known_bbox;
    point_cloud_scene_decomposer::signal signal_;
    std::vector<Eigen::Vector3f> known_object_bboxes_;
   
    const int max_distance_;
    const int min_object_size_;
   
 public:
    PointCloudClusterMatching();
    virtual void cloudCallback(
       const sensor_msgs::PointCloud2ConstPtr &);
    virtual void cloudPrevCallback(
      const sensor_msgs::PointCloud2ConstPtr &);
    virtual void indicesCallback(
      const jsk_recognition_msgs::ClusterPointIndices &);
    virtual void signalCallback(
       const point_cloud_scene_decomposer::signal &);
    virtual void manipulatedClusterCallback(
      const std_msgs::Int64 &);
    virtual void gripperEndPoseCallback(
       const geometry_msgs::PoseStamped &);
    virtual void imageCallback(
       const sensor_msgs::Image::ConstPtr &);
    virtual void imagePrevCallback(
       const sensor_msgs::Image::ConstPtr &);
    virtual void imageMaskCallback(
       const sensor_msgs::Image::ConstPtr &);
    virtual void imageMaskPrevCallback(
      const sensor_msgs::Image::ConstPtr &);
    virtual void boundingBoxCallback(
       const jsk_recognition_msgs::BoundingBoxArray &);
   
    void contourSmoothing(
       cv::Mat &);
    void clusterMovedObjectROI(
       pcl::PointCloud<PointT>::Ptr,
       pcl::PointIndices::Ptr,
       std::vector<pcl::PointIndices> &,
       const int = 64,
       const float = 0.05f);

    std::vector<pcl_msgs::PointIndices> convertToROSPointIndices(
       const std::vector<pcl::PointIndices>,
       const std_msgs::Header&);
   
    virtual void extractObjectROIIndices(
       cv::Rect_<int> &,
       pcl::PointIndices::Ptr,
       const cv::Size);
    virtual void cvMorphologicalOperations(
       const cv::Mat &, cv::Mat &, bool, int);
    virtual void getKnownObjectRegion(
       const std::vector<Eigen::Vector3f> &,
       jsk_recognition_msgs::BoundingBoxArray &,
       const float = 0.05f);



   
    virtual void objectCloudClusters(
       const pcl::PointCloud<PointT>::Ptr,
       const std::vector<pcl::PointIndices> &,
       std::vector<pcl::PointCloud<PointT>::Ptr> &);
    virtual void createImageFromObjectClusters(
       const std::vector<pcl::PointCloud<PointT>::Ptr> &,
       const sensor_msgs::CameraInfo::ConstPtr,
       const cv::Mat &,
       std::vector<cv::Mat> &,
       std::vector<cv::Rect_<int> > &);
    virtual void getObjectRegionMask(
       cv::Mat &, cv::Rect_<int> &);
    virtual void  extractKeyPointsAndDescriptors(
      const cv::Mat &image,
      cv::Mat &descriptor,
      std::vector<cv::KeyPoint> &keypoints);
    virtual void computeFeatureMatch(
       const cv::Mat &model, const cv::Mat,
       const cv::Mat &model_descriptors, const cv::Mat &,
       const std::vector<cv::KeyPoint> &,
       const std::vector<cv::KeyPoint> &scene_keypoints,
       cv::Rect_<int> &);
    cv::Rect_<int> detectMatchROI(
       const cv::Mat &, cv::Point2f &, cv::Point2f &,
       cv::Point2f &, cv::Point2f &);
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
};

#endif  //  _POINT_CLOUD_CLUSTER_MATCHING_H_
