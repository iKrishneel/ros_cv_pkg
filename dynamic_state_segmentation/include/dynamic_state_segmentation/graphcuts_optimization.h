
#ifndef _GRAPHCUTS_OPTIMIZATION_H_
#define _GRAPHCUTS_OPTIMIZATION_H_

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <dynamic_state_segmentation/maxflow/graph.h>
#include <omp.h>

class GraphCutsOptimization {
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    
 private:
    int num_threads_;
    int neigbor_size_;
    int seed_index_;
    
    std::vector<int> seed_neigbors_;
    std::vector<std::vector<int> > neigbor_indices_;
    
 protected:
    void dynamicSegmentation(const pcl::PointCloud<PointT>::Ptr,
                             const pcl::PointCloud<PointT>::Ptr);
    
 public:
    GraphCutsOptimization(const int, const std::vector<int>,
			  const std::vector<std::vector<int> >,
			  const pcl::PointCloud<PointT>::Ptr,
			  const pcl::PointCloud<PointT>::Ptr);
    
};


#endif  // _GRAPHCUTS_OPTIMIZATION_H_
