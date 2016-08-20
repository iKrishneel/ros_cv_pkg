
#pragma once
#ifndef _HANDHELD_OBJECT_REGISTRATION_H_
#define _HANDHELD_OBJECT_REGISTRATION_H_

#include <omp.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/distances.h>
#include <pcl/common/centroid.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/registration/default_convergence_criteria.h>

#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>

#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/CameraInfo.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>

#include <handheld_object_registration/transformation.h>
#include <handheld_object_registration/projected_correspondences.h>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>

namespace jsk_msgs = jsk_recognition_msgs;

#define HISTORY_WINDOW 5

class HandheldObjectRegistration {

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    typedef pcl::PointXYZRGBNormal PointNormalT;
    typedef pcl::PointCloud<PointT> PointCloud;
    typedef pcl::PointCloud<NormalT> PointNormal;

    struct Weights {
       float weight[HISTORY_WINDOW];
    };
    typedef std::vector<Weights> VoxelWeights;
   
 private:
    boost::mutex mutex_;
    boost::mutex lock_;

    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2, sensor_msgs::CameraInfo> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> sub_cinfo_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    int num_threads_;
    int min_points_size_;  //! number of points for update
   
    float weight_decay_factor_;
    float init_weight_;
   
    PointCloud::Ptr input_cloud_;
    PointNormal::Ptr input_normals_;
    pcl::PointCloud<PointNormalT>::Ptr target_points_;
    pcl::PointCloud<PointNormalT>::Ptr prev_points_;

    ProjectionMap initial_projection_;
    ProjectionMap prev_projection_;
   
    std::vector<float> point_weights_;
    VoxelWeights voxel_weights_;
   
    pcl::KdTreeFLANN<PointT>::Ptr kdtree_;
    geometry_msgs::PointStamped screen_msg_;
    bool is_init_;

    boost::shared_ptr<jsk_msgs::BoundingBox> rendering_cuboid_;
    sensor_msgs::CameraInfo::ConstPtr camera_info_;
    cv::Ptr<cv::cuda::ORB> orb_gpu_;
   
    /**
     * temp variable for dev
     */
    geometry_msgs::PoseStamped::ConstPtr pose_msg_;
    ros::Subscriber screen_pt_;
    ros::Subscriber pf_pose_;
    bool pose_flag_;
    Eigen::Affine3f prev_transform_;
    int update_counter_;
    tf::Transform previous_transform_;
    PointT prev_seed_point_;

    Eigen::Matrix4f initial_transform_;  //! cumulative transform
    std::vector<Eigen::Matrix4f> transformation_cache_;
   
 protected:
    ros::NodeHandle pnh_;
    void onInit();
    void subscribe();
    void unsubscribe();
   
    ros::Publisher pub_cloud_;
    ros::Publisher pub_icp_;
    ros::Publisher pub_bbox_;
    ros::Publisher pub_templ_;
    
 public:
    HandheldObjectRegistration();
    void cloudCB(const sensor_msgs::PointCloud2::ConstPtr &,
                 const sensor_msgs::CameraInfo::ConstPtr &);
    void screenCB(const geometry_msgs::PointStamped::ConstPtr &);
    void poseCB(const geometry_msgs::PoseStamped::ConstPtr &);
   
    bool registrationICP(pcl::PointCloud<PointNormalT>::Ptr,
                         Eigen::Matrix<float, 4, 4> &,
                         const pcl::PointCloud<PointNormalT>::Ptr,
                         const pcl::PointCloud<PointNormalT>::Ptr);
    float checkRegistrationFitness(const ProjectionMap,
                                   const pcl::PointCloud<PointNormalT>::Ptr,
                                   const ProjectionMap,
                                   const pcl::PointCloud<PointNormalT>::Ptr);
    void modelUpdate(pcl::PointCloud<PointNormalT>::Ptr,
                     pcl::PointCloud<PointNormalT>::Ptr);
    void modelVoxelUpdate(const pcl::PointCloud<PointNormalT>::Ptr,
                          const ProjectionMap,
                          const pcl::PointCloud<PointNormalT>::Ptr,
                          const ProjectionMap);
   
    bool seedRegionGrowing(PointCloud::Ptr, PointNormal::Ptr,
                           const PointT, const PointCloud::Ptr,
                           PointNormal::Ptr);
    void seedCorrespondingRegion(int *, const PointCloud::Ptr,
                                 const PointNormal::Ptr, const int, const int);
    int seedVoxelConvexityCriteria(Eigen::Vector4f, Eigen::Vector4f,
                                   Eigen::Vector4f, Eigen::Vector4f,
                                   Eigen::Vector4f, const float = 0.0f);
    void getNormals(PointNormal::Ptr, const PointCloud::Ptr);
    template<class T>
    void getPointNeigbour(std::vector<int> &,
                          const PointT, const T = 8, bool = true);
    bool project3DTo2DDepth(ProjectionMap &,
                            const pcl::PointCloud<PointNormalT>::Ptr,
                            const float = 1.0f);
    void features2D(std::vector<cv::KeyPoint> &, cv::cuda::GpuMat &,
                   const cv::Mat);
    void featureBasedTransformation(std::vector<CandidateIndices> &,
                                    const pcl::PointCloud<PointNormalT>::Ptr,
                                    const cv::Mat, const cv::Mat,
                                    const pcl::PointCloud<PointNormalT>::Ptr,
                                    const cv::Mat, const cv::Mat);
   
   
    void symmetricPlane(float *, pcl::PointCloud<PointNormalT>::Ptr,
                        const float = 0.02f);
    void plotPlane(pcl::PointCloud<PointNormalT>::Ptr, const Eigen::Vector4f,
                   const Eigen::Vector3f = Eigen::Vector3f(255, 0, 0));

    bool conditionROI(int, int, int, int, const cv::Size);
    void getPFTransformation();
};


#endif  // _HANDHELD_OBJECT_REGISTRATION_H_
