
#pragma once
#ifndef _HANDHELD_OBJECT_REGISTRATION_H_
#define _HANDHELD_OBJECT_REGISTRATION_H_

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/distances.h>
#include <pcl/common/centroid.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/features/integral_image_normal.h>

#include <geometry_msgs/PointStamped.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>
#include <omp.h>

class HandheldObjectRegistration {

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    typedef pcl::PointXYZRGBNormal PointNormalT;
    typedef pcl::PointCloud<PointT> PointCloud;
    typedef pcl::PointCloud<NormalT> PointNormal;
   
 private:
    boost::mutex mutex_;
    boost::mutex lock_;

    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2, geometry_msgs::PointStamped> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<geometry_msgs::PointStamped> screen_pt_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    int num_threads_;
    pcl::KdTreeFLANN<PointT>::Ptr kdtree_;

    // PointCloud::Ptr target_cloud_;
    // PointNormal::Ptr target_normals_;
    pcl::PointCloud<PointNormalT>::Ptr target_points_;
   
 protected:
    ros::NodeHandle pnh_;
    void onInit();
    void subscribe();
    void unsubscribe();
   
    ros::Publisher pub_cloud_;
    ros::Publisher pub_icp_;
    
 public:
    HandheldObjectRegistration();
    void cloudCB(const sensor_msgs::PointCloud2::ConstPtr &,
                 const geometry_msgs::PointStamped::ConstPtr &);
    void registrationICP(const pcl::PointCloud<PointNormalT>::Ptr,
                         Eigen::Matrix<float, 4, 4> &,
                         const pcl::PointCloud<PointNormalT>::Ptr);
    void seedRegionGrowing(PointCloud::Ptr, PointNormal::Ptr,
                           const geometry_msgs::PointStamped,
                           const PointCloud::Ptr, PointNormal::Ptr);
    void seedCorrespondingRegion(int *, const PointCloud::Ptr,
                                 const PointNormal::Ptr, const int, const int);
    int seedVoxelConvexityCriteria(Eigen::Vector4f, Eigen::Vector4f,
                                   Eigen::Vector4f, Eigen::Vector4f,
                                   Eigen::Vector4f, const float = 0.0f);
    void getNormals(PointNormal::Ptr, const PointCloud::Ptr);
    template<class T>
    void getPointNeigbour(std::vector<int> &,
                          const PointCloud::Ptr,
                          const PointT, const T = 8, bool = true);
   
};


#endif  // _HANDHELD_OBJECT_REGISTRATION_H_
