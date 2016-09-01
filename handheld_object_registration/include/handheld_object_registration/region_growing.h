
#ifndef _REGION_GROWING_H_
#define _REGION_GROWING_H_

#include <omp.h>

#include <ros/ros.h>
#include <ros/console.h>

#include <pcl/kdtree/kdtree_flann.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>

#include <opencv2/opencv.hpp>

class RegionGrowing {

 private:
    typedef pcl::PointXYZRGBNormal PointNormalT;
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    typedef pcl::PointCloud<PointT> PointCloud;
    typedef pcl::PointCloud<NormalT> PointNormal;
   
    template<class T>
    void getPointNeigbour(std::vector<int> &,
                          const PointT, const T = 8, bool = true);

 protected:
    pcl::KdTreeFLANN<PointT>::Ptr kdtree_;
    sensor_msgs::CameraInfo::ConstPtr camera_info_;
    int num_threads_;

 public:
    RegionGrowing(const sensor_msgs::CameraInfo::ConstPtr, const int = 8);
    bool seedRegionGrowing(pcl::PointCloud<PointNormalT>::Ptr,
                           const PointT, const PointCloud::Ptr,
                           PointNormal::Ptr);
    void seedCorrespondingRegion(int *, const PointCloud::Ptr,
                                 const PointNormal::Ptr,
                                 const int, const int);
    int seedVoxelConvexityCriteria(Eigen::Vector4f, Eigen::Vector4f,
                                   Eigen::Vector4f, Eigen::Vector4f,
                                   Eigen::Vector4f, const float = 0.0f);

    void fastSeedRegionGrowing(pcl::PointCloud<PointNormalT>::Ptr,
                               cv::Point2i &, const PointCloud::Ptr,
                               const PointNormal::Ptr, const PointT);
   

};



#endif /* _REGION_GROWING_H_ */

