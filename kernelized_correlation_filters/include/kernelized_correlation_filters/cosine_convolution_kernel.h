
#pragma once
#ifndef _COSINE_CONVOLUTION_KERNEL_H_
#define _COSINE_CONVOLUTION_KERNEL_H_

#include <kernelized_correlation_filters/cuda_common.h>

#include <stdio.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cublas.h>
#include <cublas_v2.h>

#define GRID_SIZE 16
#define CNN_FILTER_SIZE 256



#endif // _COSINE_CONVOLUTION_KERNEL_H_

