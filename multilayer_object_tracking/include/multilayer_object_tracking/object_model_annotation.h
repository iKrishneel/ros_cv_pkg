// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo, Japan

#ifndef _OBJECT_MODEL_ANNOTATION_H_
#define _OBJECT_MODEL_ANNOTATION_H_

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
// #include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/video/background_segm.hpp>

#include <boost/thread/mutex.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/distances.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PoseStamped.h>

class ObjectModelAnnotation {
 private:
    typedef pcl::PointXYZRGB PointT;
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::Image,
       sensor_msgs::PointCloud2,
       geometry_msgs::PolygonStamped> SyncPolicy;
   
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::Image> sub_image_;
    message_filters::Subscriber<geometry_msgs::PolygonStamped> sub_screen_pt_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    ros::Publisher pub_cloud_;
    ros::Publisher pub_background_;
    ros::Publisher pub_image_;
    ros::Publisher pub_pose_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
   
 public:
    ObjectModelAnnotation();
    virtual void callback(
       const sensor_msgs::Image::ConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &,
       const geometry_msgs::PolygonStampedConstPtr &);
    void imageToPointCloudIndices(
       pcl::PointCloud<PointT>::Ptr,
       pcl::PointIndices::Ptr,
       const cv::Size, const cv::Rect_<int>);
    void getAnnotatedObjectCloud(
       pcl::PointCloud<PointT>::Ptr,
       const cv::Mat &,
       const cv::Rect_<int>);
    void compute3DCentroids(
       const pcl::PointCloud<PointT>::Ptr,
       Eigen::Vector4f &);
    void backgroundPointCloudIndices(
        pcl::PointCloud<PointT>::Ptr,
        const pcl::PointCloud<PointT>::Ptr,
        const Eigen::Vector4f,
        const cv::Size, const cv::Rect_<int>);
    float templateCloudFilterLenght(
       const pcl::PointCloud<PointT>::Ptr,
       const Eigen::Vector4f);
    bool filterPointCloud(
       pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<PointT>::Ptr,
       const Eigen::Vector4f,
       const float = 2.0f);
};


#endif  //_OBJECT_MODEL_ANNOTATION_H_
