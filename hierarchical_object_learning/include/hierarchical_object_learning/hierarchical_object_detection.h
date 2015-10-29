
#ifndef _HIERARCHICAL_OBJECT_DETECTION_H_
#define _HIERARCHICAL_OBJECT_DETECTION_H_

#include <hierarchical_object_learning/hierarchical_object_learning.h>

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

#include <rosbag/bag.h>
#include <rosbag/view.h>

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
#include <pcl/filters/voxel_grid.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PolygonStamped.h>
#include <std_msgs/Header.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/PointsArray.h>
#include <jsk_recognition_msgs/Histogram.h>

#include <multilayer_object_tracking/ReferenceModelBundle.h>
#include <hierarchical_object_learning/FeatureArray.h>
#include <hierarchical_object_learning/FitFeatureModel.h>

#include <omp.h>

class HierarchicalObjectDetection : public HierarchicalObjectLearning {
 private:
    typedef pcl::PointXYZRGB PointT;
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::CameraInfo,
       sensor_msgs::Image,
       sensor_msgs::PointCloud2> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> sub_info_;
    message_filters::Subscriber<sensor_msgs::Image> sub_image_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    ros::Publisher pub_cloud_;
    ros::Publisher pub_image_;
    ros::Publisher pub_pose_;
    ros::ServiceClient predictor_client_;

    int num_threads_;

    std::string source_type_;
    int cluster_size_;
    int min_cloud_size_;
    int neigbour_size_;
    float downsize_;
   
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
  
  public:
    HierarchicalObjectDetection();
  
    virtual void callback(
        const sensor_msgs::CameraInfo::ConstPtr &,
        const sensor_msgs::Image::ConstPtr &,
        const sensor_msgs::PointCloud2::ConstPtr &);
  
};


#endif   // _HIERARCHICAL_OBJECT_DETECTION_H_
