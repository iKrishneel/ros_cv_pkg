
#ifndef _CUBOID_BILATERAL_SYMMETRIC_SEGMENTATION_H_
#define _CUBOID_BILATERAL_SYMMETRIC_SEGMENTATION_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/distances.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>

#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_utils/geo/polygon.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>
#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <boost/thread/mutex.hpp>

namespace jsk_msgs = jsk_recognition_msgs;

class CuboidBilateralSymmetricSegmentation {
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
   
 private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
  
    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2,
       // sensor_msgs::PointCloud2,
       jsk_msgs::ClusterPointIndices,
       jsk_msgs::BoundingBoxArray> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_normal_;
    message_filters::Subscriber<jsk_msgs::ClusterPointIndices> sub_indices_;
    message_filters::Subscriber<jsk_msgs::BoundingBoxArray> sub_boxes_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

 protected:
    void onInit();
    void subscribe();
    void unsubscribe();

    ros::Publisher pub_cloud_;
    ros::Publisher pub_edge_;
    ros::Publisher pub_indices_;
    ros::ServiceClient srv_client_;
   
 public:
    CuboidBilateralSymmetricSegmentation();
    void cloudCB(const sensor_msgs::PointCloud2::ConstPtr &,
                 const jsk_msgs::ClusterPointIndices::ConstPtr &,
                 const jsk_msgs::BoundingBoxArray::ConstPtr &);
    void boundingBoxSymmetricalAxisPlane(
       std::vector<pcl::ModelCoefficients::Ptr> &,
       const jsk_msgs::BoundingBoxArray::ConstPtr &);
};


#endif  // _CUBOID_BILATERAL_SYMMETRIC_SEGMENTATION_H_
