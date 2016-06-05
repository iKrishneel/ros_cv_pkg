
#ifndef _CUBOID_BILATERAL_SYMMETRIC_SEGMENTATION_H_
#define _CUBOID_BILATERAL_SYMMETRIC_SEGMENTATION_H_

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

#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Header.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_utils/geo/polygon.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>
#include <omp.h>

#include <cuboid_bilateral_symmetric_segmentation/oriented_bounding_box.h>
#include <cuboid_bilateral_symmetric_segmentation/maxflow/graph.h>
#include <cuboid_bilateral_symmetric_segmentation/object_region_handler.h>

class CuboidBilateralSymmetricSegmentation:
    public SupervoxelSegmentation,
    public OrientedBoundingBox {
   
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    typedef pcl::PointXYZRGBNormal PointNormalT;
    typedef jsk_msgs::ModelCoefficientsArrayConstPtr ModelCoefficients;
    typedef std::map<uint32_t, int> UInt32Map;
    typedef std::map<uint32_t, std::vector<uint32_t> > AdjacentList;
    typedef pcl::VoxelGridOcclusionEstimation<PointT> OcclusionHandler;
    typedef Graph<float, float, float> GraphType;
   
 private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
  
    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2, sensor_msgs::PointCloud2,
       sensor_msgs::PointCloud2, jsk_msgs::PolygonArray,
       jsk_msgs::ModelCoefficientsArray> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_prob_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_normal_;
    message_filters::Subscriber<jsk_msgs::PolygonArray> sub_planes_;
    message_filters::Subscriber<jsk_msgs::ModelCoefficientsArray> sub_coef_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    int num_threads_;
    uint32_t min_cluster_size_;
    float leaf_size_;
    float symmetric_angle_thresh_;
    double neigbor_dist_thresh_;  //! distance for point to be symm
   
    boost::shared_ptr<OcclusionHandler> occlusion_handler_;
    pcl::KdTreeFLANN<PointT>::Ptr kdtree_;
    std_msgs::Header header_;
    SupervoxelMap convex_supervoxel_clusters_;
    PointNormalT seed_info_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
   
    ros::Publisher pub_cloud_;
    ros::Publisher pub_edge_;
    ros::Publisher pub_indices_;
    ros::Publisher pub_bbox_;
    ros::Publisher pub_normal_;
    ros::Publisher pub_object_;
   
 public:
    CuboidBilateralSymmetricSegmentation();
    void cloudCB(const sensor_msgs::PointCloud2::ConstPtr &,
                 const sensor_msgs::PointCloud2::ConstPtr &,
                 const sensor_msgs::PointCloud2::ConstPtr &,
                 const jsk_msgs::PolygonArrayConstPtr &,
                 const ModelCoefficients &);
    void segmentation(pcl::PointIndices::Ptr,
                      pcl::PointCloud<PointT>::Ptr,
                      const jsk_msgs::PolygonArrayConstPtr &,
                      const ModelCoefficients &);
    void supervoxelDecomposition(SupervoxelMap &,
                                 pcl::PointCloud<NormalT>::Ptr,
                                 const pcl::PointCloud<PointT>::Ptr);
    bool mergeNeigboringSupervoxels(SupervoxelMap &, AdjacencyList &,
                                    const int);
    float coplanarityCriteria(const Eigen::Vector4f, const Eigen::Vector4f,
                              const Eigen::Vector4f, const Eigen::Vector4f,
                              const float = 10, const float = 0.02f);
    void updateSupervoxelClusters(SupervoxelMap &,
                                 const uint32_t, const uint32_t);
    void supervoxelAdjacencyList(AdjacencyList &, const SupervoxelMap);
    void supervoxel3DBoundingBox(jsk_msgs::BoundingBox &,
                                 pcl::PointCloud<PointT>::Ptr,
                                 pcl::PointCloud<NormalT>::Ptr,
                                 const SupervoxelMap &,
                                 const jsk_msgs::PolygonArrayConstPtr &,
                                 const ModelCoefficients &,
                                 const int);
    void symmetryBasedObjectHypothesis(SupervoxelMap &,
                                       pcl::PointIndices::Ptr,
                                       const pcl::PointCloud<PointT>::Ptr,
                                       const jsk_msgs::PolygonArrayConstPtr &,
                                       const ModelCoefficients &);
    bool optimizeSymmetricalPlane(Eigen::Vector4f &,
                                  pcl::PointCloud<PointT>::Ptr);
    bool symmetricalConsistency(Eigen::Vector4f &, float &,
                                pcl::PointCloud<PointT>::Ptr,
                                pcl::PointCloud<NormalT>::Ptr,
                                const pcl::PointCloud<PointT>::Ptr,
                                const jsk_msgs::BoundingBox);
    float symmetricalPlaneEnergy(pcl::PointCloud<PointT>::Ptr,
                                 const pcl::PointCloud<NormalT>::Ptr,
                                 const pcl::PointCloud<PointT>::Ptr,
                                 const int, const std::vector<Eigen::Vector4f>);
    void symmetricalShapeMap(pcl::PointCloud<PointT>::Ptr,
                             const pcl::PointCloud<PointT>::Ptr,
                             const Eigen::Vector4f);
   
    bool minCutMaxFlow(pcl::PointCloud<PointT>::Ptr,
                       pcl::PointCloud<NormalT>::Ptr,
                       pcl::PointIndices::Ptr,
                       const Eigen::Vector4f);
   
    bool occlusionRegionCheck(const PointT);
    template<class T>
    void getPointNeigbour(std::vector<int> &, const PointT,
                          const T = 8, bool = true);
    template<class T>
    void estimateNormals(const pcl::PointCloud<PointT>::Ptr,
                         pcl::PointCloud<NormalT>::Ptr,
                         T = 0.05f, bool = false) const;
    int seedVoxelConvexityCriteria(Eigen::Vector4f, Eigen::Vector4f,
                                   Eigen::Vector4f, Eigen::Vector4f,
                                   const float = 0.0f);
    pcl::PointXYZRGBNormal convertVector4fToPointXyzRgbNormal(
       const Eigen::Vector3f &, const Eigen::Vector3f &,
       const Eigen::Vector3f);

   
    struct SortVector {
       bool operator() (int i, int j) {
         return (i < j);
      }
    } sortVector;
};

#endif  // _CUBOID_BILATERAL_SYMMETRIC_SEGMENTATION_H_
