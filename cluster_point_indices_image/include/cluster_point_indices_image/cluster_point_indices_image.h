
#ifndef _CLUSTER_POINT_INDICES_IMAGE_H_
#define _CLUSTER_POINT_INDICES_IMAGE_H_

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

#include <jsk_recognition_msgs/ClusterPointIndices.h>

class ClusterPointIndicesToImage {
 private:
    typedef pcl::PointXYZRGB PointT;
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
    typedef  message_filters::sync_policies::ApproximateTime<
       jsk_recognition_msgs::ClusterPointIndices,
       sensor_msgs::PointCloud2> SyncPolicy;

    message_filters::Subscriber<
      jsk_recognition_msgs::ClusterPointIndices> sub_indices_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_select_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    ros::Publisher pub_cloud_;
    ros::Publisher pub_image_;

 protected:
    void onInit();
    void subscribe();
void unsubscribe();

 public:
    ClusterPointIndicesToImage();
    virtual void callback(
       const jsk_recognition_msgs::ClusterPointIndicesConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &);
    virtual std::vector<pcl::PointIndices::Ptr>
    clusterPointIndicesToPointIndices(
       const jsk_recognition_msgs::ClusterPointIndicesConstPtr &);
};


#endif  // _CLUSTER_POINT_INDICES_IMAGE_H_
