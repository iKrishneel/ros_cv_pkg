
#ifndef _TRANSFORMATION_H_
#define _TRANSFORMATION_H_

#include <stdio.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cublas.h>
#include <cublas_v2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZRGBNormal PointTYPE;

#define GRID_SIZE 16

void transformPointCloudWithNormalsGPU(pcl::PointCloud<PointTYPE>::Ptr,
                                       pcl::PointCloud<PointTYPE>::Ptr,
                                       const Eigen::Matrix4f);
   

#endif /* _TRANSFORMATION_H_ */
