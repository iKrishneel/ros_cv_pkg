
#ifndef _DYNAMIC_STATE_SEGMENTATION_H_
#define _DYNAMIC_STATE_SEGMENTATION_H_

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

#include <dynamic_state_segmentation/Feature3DClustering.h>
#include <dynamic_state_segmentation/maxflow/graph.h>
#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <boost/thread/mutex.hpp>

class DynamicStateSegmentation {

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    typedef pcl::FPFHSignature33 FPFH;
    typedef Graph<float, float, float> GraphType;
    
    struct SortVector {
       bool operator() (int i, int j) {
          return (i < j);
       }
    } sortVector;

#define HARD_SET_WEIGHT 255
    
 private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
  
    typedef  message_filters::sync_policies::ApproximateTime<
        sensor_msgs::PointCloud2,
        geometry_msgs::PointStamped> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<geometry_msgs::PointStamped> screen_pt_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    PointT seed_point_;
    NormalT seed_normal_;
    int seed_index_;

    int num_threads_;
    int neigbor_size_;
    int min_cluster_size_;
    
    pcl::KdTreeFLANN<PointT>::Ptr kdtree_;

 protected:
    void onInit();
    void subscribe();
    void unsubscribe();

    ros::Publisher pub_cloud_;
    ros::Publisher pub_edge_;
    ros::Publisher pub_indices_;
    ros::ServiceClient srv_client_;
    
 public:
    DynamicStateSegmentation();
    virtual void cloudCB(const sensor_msgs::PointCloud2::ConstPtr &,
                         const geometry_msgs::PointStamped::ConstPtr &);

   /**
    * seeded region growing
    */
    void seedCorrespondingRegion(std::vector<int> &,
                                 const pcl::PointCloud<PointT>::Ptr,
                                 const pcl::PointCloud<NormalT>::Ptr,
                                 const int);
    template<class T>
    void getPointNeigbour(std::vector<int> &,
                          const pcl::PointCloud<PointT>::Ptr,
                          const PointT, const T = 8, bool = true);
    int seedVoxelConvexityCriteria(Eigen::Vector4f, Eigen::Vector4f,
                                    Eigen::Vector4f, Eigen::Vector4f,
                                    const float = 0.0f, bool = true);
    template<class T>
    void estimateNormals(const pcl::PointCloud<PointT>::Ptr,
                         pcl::PointCloud<NormalT>::Ptr,
                         T = 0.05f, bool = false) const;
    void regionOverSegmentation(pcl::PointCloud<PointT>::Ptr,
                                pcl::PointCloud<NormalT>::Ptr,
                                const pcl::PointCloud<PointT>::Ptr,
                                const pcl::PointCloud<NormalT>::Ptr);
    std::vector<Eigen::Vector4f>
    doEuclideanClustering(std::vector<pcl::PointIndices> &,
			  const pcl::PointCloud<PointT>::Ptr,
			  const pcl::PointIndices::Ptr, bool = false,
			  const float = 0.02f,const int = 50, const int = 20000);
    bool extractSeededCloudCluster(pcl::PointCloud<PointT>::Ptr);
    /**
     * functions for CRF
     */
    void dynamicSegmentation(pcl::PointCloud<PointT>::Ptr,
                             pcl::PointCloud<PointT>::Ptr,
                             const pcl::PointCloud<NormalT>::Ptr);
    void potentialFunctionKernel(std::vector<std::vector<int> > &,
                                 pcl::PointCloud<PointT>::Ptr,
                                 const pcl::PointCloud<PointT>::Ptr,
                                 const pcl::PointCloud<NormalT>::Ptr);
    void normalEdge(pcl::PointCloud<PointT>::Ptr,
                    pcl::PointCloud<PointT>::Ptr,
                    const pcl::PointCloud<PointT>::Ptr,
                    const pcl::PointCloud<NormalT>::Ptr);
    int localVoxelConvexityCriteria(Eigen::Vector4f, Eigen::Vector4f,
                                    Eigen::Vector4f, const float = 0.0f);
    void edgeBoundaryOutlierFiltering(pcl::PointCloud<PointT>::Ptr,
				      pcl::PointIndices::Ptr,
				      const float = 0.01f, const int = 100);
   /**
    * function for saliency term
    */
    void clusterFeatures(std::vector<pcl::PointIndices> &,
                         pcl::PointCloud<PointT>::Ptr,
                         const pcl::PointCloud<NormalT>::Ptr,
                         const int, const float);
    void mergeVoxelClusters(
       const dynamic_state_segmentation::Feature3DClustering,
       pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<NormalT>::Ptr,
       const std::vector<std::vector<int> >);
   
    void computeFeatures(pcl::PointCloud<PointT>::Ptr,
                         const pcl::PointCloud<NormalT>::Ptr, const int);
    
    void computeFPFH(pcl::PointCloud<FPFH>::Ptr,
                     const pcl::PointCloud<PointT>::Ptr,
                     const pcl::PointCloud<pcl::Normal>::Ptr,
                     const float = 0.05f) const;
    template<class T>
    T histogramKernel(const FPFH, const FPFH, const int);
    
    
    void pointColorContrast(pcl::PointCloud<PointT>::Ptr,
                            const pcl::PointCloud<PointT>::Ptr, const int);

    // distances
    template<class T>
    T intensitySimilarityMetric(const PointT, const PointT, const bool = false);
    template<class T>
    T distancel2(const Eigen::Vector3f, const Eigen::Vector3f, bool = false);
};
#endif  // _DYNAMIC_STATE_SEGMENTATION_H_
