
#pragma once
#ifndef _PROJECTED_CORRESPONDENCES_H_
#define _PROJECTED_CORRESPONDENCES_H_

#include <omp.h>
#include <stdio.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cublas.h>
#include <cublas_v2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_estimation.h>

#include <handheld_object_registration/data_structure.h>
#include <opencv2/opencv.hpp>

typedef pcl::PointXYZRGBNormal PointTYPE;

template<class T, int N> struct __align__(16) cuMat{
    T data[N];
};

__host__ __device__ struct Correspondence {
    int query_index;
    int match_index;
    float distance;

    Correspondence(int q, int m, float d) :
       query_index(q), match_index(m), distance(d){}
};

#define GRID_SIZE 16
#define DISTANCE_THRESH  0.02f

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480

const int NUMBER_OF_ELEMENTS = 3;
const int POINT_ELEMENTS = 12;

void cudaAssert(cudaError_t, char *, int, bool = true);
__host__ __device__ __align__(16)
    int cuDivUp(int, int);

__global__ __forceinline__
void findCorrespondencesGPU(Correspondence *,
                            cuMat<float, NUMBER_OF_ELEMENTS> *,
                            int *, cuMat<float, NUMBER_OF_ELEMENTS> *,
                            int *, const int, const int, const int);

bool allocateCopyDataToGPU(pcl::Correspondences &, float &, bool,
                           const pcl::PointCloud<PointTYPE>::Ptr,
                           const ProjectionMap &,
                           const pcl::PointCloud<PointTYPE>::Ptr,
                           const ProjectionMap &);


bool allocateCopyDataToGPU2(pcl::Correspondences &, float &, bool,
                           const pcl::PointCloud<PointTYPE>::Ptr,
                           const ProjectionMap &,
                           const pcl::PointCloud<PointTYPE>::Ptr,
                           const ProjectionMap &);

void cudaGlobalAllocFree();



#endif /* _PROJECTED_CORRESPONDENCES_H_ */
