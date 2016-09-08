
#ifndef _EVALUATE_POINT_CLOUD_ACCURACY_H_
#define _EVALUATE_POINT_CLOUD_ACCURACY_H_

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/distances.h>
#include <pcl/common/centroid.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>

class EvaluateAccuracy {

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    typedef pcl::PointCloud<PointT> PointCloud;

 private:
    float DISTANCE_THRESH_;
    std::string write_path_;
   
 public:
    EvaluateAccuracy(const std::string, const std::string,
                     const std::string);
    void readDataFromFile(std::vector<std::string> &,
                          const std::string);
    void evaluate(std::vector<std::string>,
                  std::vector<std::string>);
    void accuracy(int &, int &, const PointCloud::Ptr,
                 const pcl::KdTreeFLANN<PointT>::Ptr);
    template<class T>
    void getPointNeigbour(std::vector<int> &, std::vector<float> &,
                          const pcl::KdTreeFLANN<PointT>::Ptr,
                          const PointT, const T, bool = true);
};


#endif /* _EVALUATE_POINT_CLOUD_ACCURACY_H_ */
