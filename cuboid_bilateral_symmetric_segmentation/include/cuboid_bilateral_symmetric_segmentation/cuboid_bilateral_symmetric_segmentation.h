
#ifndef _CUBOID_BILATERAL_SYMMETRIC_SEGMENTATION_H_
#define _CUBOID_BILATERAL_SYMMETRIC_SEGMENTATION_H_


#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl_conversions/pcl_conversions.h>
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
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>

#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Header.h>

#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_utils/geo/polygon.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>
#include <omp.h>

#include <cuboid_bilateral_symmetric_segmentation/supervoxel_segmentation.h>
#include <cuboid_bilateral_symmetric_segmentation/oriented_bounding_box.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <boost/thread/mutex.hpp>

class CuboidBilateralSymmetricSegmentation:
    public SupervoxelSegmentation,
    public OrientedBoundingBox {
   
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    typedef pcl::PointXYZRGBNormal PointNormalT;
    typedef pcl::SupervoxelClustering<PointT>::VoxelAdjacencyList AdjacencyList;

    typedef std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> SupervoxelMap;
    typedef std::map<uint32_t, int> UInt32Map;
    typedef pcl::VoxelGridOcclusionEstimation<PointT> OcclusionHandler;
    // typedef pcl::VoxelGridOcclusionEstimation<PointNormalT> OcclusionHandler;
   
 private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
  
    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2,
       sensor_msgs::PointCloud2,
       jsk_msgs::PolygonArray,
       jsk_msgs::ModelCoefficientsArray> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_normal_;
    message_filters::Subscriber<jsk_msgs::PolygonArray> sub_planes_;
    message_filters::Subscriber<jsk_msgs::ModelCoefficientsArray> sub_coef_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    uint32_t min_cluster_size_;
    float leaf_size_;
    boost::shared_ptr<OcclusionHandler> occlusion_handler_;
    pcl::KdTreeFLANN<PointT>::Ptr kdtree_;

    std_msgs::Header header_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();

    ros::Publisher pub_cloud_;
    ros::Publisher pub_edge_;
    ros::Publisher pub_indices_;
    ros::Publisher pub_bbox_;
    ros::Publisher pub_normal_;

   
 public:
    CuboidBilateralSymmetricSegmentation();
    void cloudCB(const sensor_msgs::PointCloud2::ConstPtr &,
                 const sensor_msgs::PointCloud2::ConstPtr &,
                 const jsk_msgs::PolygonArrayConstPtr &,
                 const jsk_msgs::ModelCoefficientsArrayConstPtr &);
    void supervoxelDecomposition(SupervoxelMap &,
                                 pcl::PointCloud<NormalT>::Ptr,
                                 const pcl::PointCloud<PointT>::Ptr);
    float coplanarityCriteria(const Eigen::Vector4f, const Eigen::Vector4f,
                              const Eigen::Vector4f, const Eigen::Vector4f,
                              const float = 10, const float = 0.02f);
    void updateSupervoxelClusters(SupervoxelMap &,
                                 const uint32_t, const uint32_t);
    void supervoxel3DBoundingBox(jsk_msgs::BoundingBox &, const SupervoxelMap &,
                                 const jsk_msgs::PolygonArrayConstPtr &,
                                 const jsk_msgs::ModelCoefficientsArrayConstPtr &,
                                 const int);
    bool symmetricalConsistency(pcl::PointCloud<PointT>::Ptr,
                                pcl::PointCloud<NormalT>::Ptr,
                                const jsk_msgs::BoundingBox);
    bool occlusionRegionCheck(const PointT);
    template<class T>
    void getPointNeigbour(std::vector<int> &, const PointT,
                          const T = 8, bool = true);

    pcl::PointXYZRGBNormal convertVector4fToPointXyzRgbNormal(
       const Eigen::Vector3f &, const Eigen::Vector3f &,
       const Eigen::Vector3f);
};


#endif  // _CUBOID_BILATERAL_SYMMETRIC_SEGMENTATION_H_
