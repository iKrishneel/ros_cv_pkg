
#ifndef _OBJECT_REGION_HANDLER_H_
#define _OBJECT_REGION_HANDLER_H_

#include <omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/distances.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>

#include <std_msgs/Header.h>

#include <cuboid_bilateral_symmetric_segmentation/supervoxel_segmentation.h>

class ObjectRegionHandler: public SupervoxelSegmentation {

    typedef pcl::Normal NormalT;
    typedef std::map<uint32_t, int> UInt32Map;
   
    struct IndicesMap {
       uint32_t label;
       int index;
    };
   
 private:
    pcl::PointCloud<PointT>::Ptr in_cloud_;
    pcl::PointCloud<NormalT>::Ptr in_normals_;

    AdjacencyList adjacency_list_;
    SupervoxelMap supervoxel_clusters_;
    std::vector<IndicesMap> indices_map_;
    pcl::PointIndices::Ptr region_indices_;  //! indices of selected points
   
    PointT seed_point_;
    NormalT seed_normal_;
   
    int min_cluster_size_;
    int num_threads_;
    int seed_index_;
    int neigbor_size_;
    uint32_t prev_index_;
    Eigen::Vector4f origin_;
   
    std_msgs::Header header_;
    std::vector<pcl::PointIndices> all_indices_;
   
    int seedVoxelConvexityCriteria(Eigen::Vector4f, Eigen::Vector4f,
                                   Eigen::Vector4f, Eigen::Vector4f,
                                   const float = 0.0f, bool = true);
    template<class T>
    void estimateNormals(const pcl::PointCloud<PointT>::Ptr,
                         pcl::PointCloud<NormalT>::Ptr,
                         T = 0.05f, bool = false) const;
    template<class T>
     void pointNeigbour(std::vector<int> &, const PointT,
                        const T = 8, bool = true);
    void seedCorrespondingRegion(std::vector<int> &,
                                 const pcl::PointCloud<PointT>::Ptr,
                                 const pcl::PointCloud<NormalT>::Ptr,
                                 const int);
    void regionOverSegmentation(pcl::PointCloud<PointT>::Ptr,
                                pcl::PointCloud<NormalT>::Ptr,
                                const pcl::PointCloud<PointT>::Ptr,
                                const pcl::PointCloud<NormalT>::Ptr);
    void doEuclideanClustering(std::vector<pcl::PointIndices> &,
                               const pcl::PointCloud<PointT>::Ptr,
                               const pcl::PointIndices::Ptr,
                               const float = 0.02f, const int = 50,
                               const int = 20000);
    void getRegionSupervoxels(SupervoxelMap &, pcl::PointCloud<PointT>::Ptr);
    void supervoxelCoplanarityMerge(SupervoxelMap &, AdjacencyList &);
    void updateSupervoxelClusters(SupervoxelMap &, const uint32_t,
                                  const uint32_t);
    float coplanarityCriteria(const Eigen::Vector4f, const Eigen::Vector4f,
                              const Eigen::Vector4f, const Eigen::Vector4f,
                              const float, const float);
   
 protected:
    pcl::KdTreeFLANN<PointT>::Ptr kdtree_;
   
 public:
    ObjectRegionHandler(const int = 50, const int = 1);
    bool setInputCloud(const pcl::PointCloud<PointT>::Ptr, std_msgs::Header);
    bool getCandidateRegion(SupervoxelMap &, pcl::PointCloud<PointT>::Ptr,
                            pcl::PointXYZRGBNormal &);
    void updateObjectRegion(pcl::PointCloud<PointT>::Ptr,
                            const pcl::PointIndices::Ptr);
    void getLabels(std::vector<pcl::PointIndices> &);
       
    pcl::PointCloud<PointT>::Ptr sv_cloud_;
};

#endif  // _OBJECT_REGION_HANDLER_H_
