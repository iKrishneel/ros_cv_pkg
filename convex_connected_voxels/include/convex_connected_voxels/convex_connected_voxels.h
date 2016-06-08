
#ifndef _CONVEX_CONNECTED_VOXEL_H_
#define _CONVEX_CONNECTED_VOXEL_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <dynamic_reconfigure/server.h>

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

#include <convex_connected_voxels/supervoxel_segmentation.h>

class ConvexConnectedVoxels: public SupervoxelSegmentation {

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::SupervoxelClustering<PointT>::VoxelAdjacencyList AdjacencyList;
   
 private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
    typedef message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2,
       jsk_recognition_msgs::PolygonArray> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<jsk_recognition_msgs::PolygonArray> sub_normal_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    ros::Publisher pub_cloud_;
    ros::Publisher pub_indices_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
   
 public:
    ConvexConnectedVoxels();
    virtual void callback(
       const sensor_msgs::PointCloud2::ConstPtr &,
       const jsk_recognition_msgs::PolygonArray::ConstPtr &);
    virtual void surfelLevelObjectHypothesis(
       pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<pcl::Normal>::Ptr,
       std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > &);
    void updateSupervoxelClusters(
       std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> &,
       const uint32_t, const uint32_t);
    virtual Eigen::Vector4f cloudMeanNormal(
       const pcl::PointCloud<pcl::Normal>::Ptr, bool = false);
   
};
#endif   // _CONVEX_CONNECTED_VOXEL_H_

