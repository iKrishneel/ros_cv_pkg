
#ifndef _DYNAMIC_STATE_SEGMENTATION_H_
#define _DYNAMIC_STATE_SEGMENTATION_H_

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

class DynamicStateSegmentation {
  typedef pcl::PointXYZRGB PointT;
 private:
  boost::mutex mutex_;
  ros::NodeHandle pnh_;
  
  typedef  message_filters::sync_policies::ApproximateTime<
    sensor_msgs::PointCloud2,
    sensor_msgs::PointCloud2> SyncPolicy;
  // message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
  // boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;
  
 protected:
  void onInit();
  void subscribe();
  void unsubscribe();
  
 public:
  DynamicStateSegmentation();
  virtual void cloudCB(const sensor_msgs::PointCloud2::ConstPtr &);
  
};


#endif  // _DYNAMIC_STATE_SEGMENTATION_H_
