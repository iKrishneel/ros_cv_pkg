
#ifndef _POINT_CLOUD_OBJECT_DETECTION_H_
#define _POINT_CLOUD_OBJECT_DETECTION_H_

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_pcl_ros/pcl_conversion_util.h>
#include <jsk_recognition_msgs/RectArray.h>

#include <ros/ros.h>
#include <ros/console.h>

// ROS sensor message header directives
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

// PCL header directives
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>
#include <pcl/common/impl/common.hpp>
#include <pcl/registration/distances.h>

#include <jsk_pcl_ros/EuclideanSegment.h>

#include <iostream>
#include <vector>

class PointCloudObjectDetection {

 public:
    PointCloudObjectDetection();
   
    typedef pcl::PointXYZRGB PointT;

 private:
    virtual void subscribe();
    virtual void unsubscribe();
    virtual void cloudCallback(
       const sensor_msgs::PointCloud2::ConstPtr&);
    virtual void jskRectArrayCb(
       const jsk_recognition_msgs::RectArray &);
   
    
    void extractPointCloudIndicesFromJSKRect(
       jsk_recognition_msgs::Rect, pcl::PointIndices::Ptr);

   
    ros::NodeHandle pnh_;
    boost::mutex mutex_;
    ros::Subscriber sub_cloud_;
    ros::Subscriber sub_rects_;
   
    ros::Publisher pub_cloud_;
   
   
    // **use message filters *** 
    int cloud_width;
    int cloud_height;

    pcl::PointIndices::Ptr filter_indices_;
};

#endif  // _POINT_CLOUD_OBJECT_DETECTION_H_
