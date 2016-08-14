
#pragma once
#ifndef _PROJECTED_CORRESPONDENCES_H_
#define _PROJECTED_CORRESPONDENCES_H_

#include <stdio.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cublas.h>
#include <cublas_v2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <handheld_object_registration/data_structure.h>
#include <opencv2/opencv.hpp>


typedef pcl::PointXYZRGBNormal PointTYPE;


template<class T, int N> struct __align__(16) cuMat{
    T data[N];
};

__host__ __device__ struct Correspondence {
    int query_index;
    int match_index;
};

#define GRID_SIZE 16
#define DISTANCE_THRESH  0.05f
#define IMAGE_SIZE 640 * 480

const int NUMBER_OF_ELEMENTS = 3;

void cudaAssert(cudaError_t, char *, int, bool = true);
__host__ __device__ __align__(16)
    int cuDivUp(int, int);
// __host__ __device__ __forceinline__
// void cuConditionROI(cuRect *, int, int);

__global__ __forceinline__
void findCorrespondencesGPU(Correspondence *,
                            cuMat<float, NUMBER_OF_ELEMENTS> *,
                            int *,
                            cuMat<float, NUMBER_OF_ELEMENTS> *,
                            // cuMat<int, 2> *,
                            int *,
                            const int, const int,
                            const int, const int);

bool allocateCopyDataToGPU(bool,
                           const pcl::PointCloud<PointTYPE>::Ptr,
                           const ProjectionMap &,
                           const pcl::PointCloud<PointTYPE>::Ptr,
                           const ProjectionMap &);
   
void estimatedCorrespondences(bool,
                              const pcl::PointCloud<PointTYPE>::Ptr,
                              const ProjectionMap &,
                              const pcl::PointCloud<PointTYPE>::Ptr,
                              const ProjectionMap &);

void cudaGlobalAllocFree();



#endif /* _PROJECTED_CORRESPONDENCES_H_ */
