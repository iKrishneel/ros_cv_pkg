
#pragma once
#ifndef _DISCRETE_FOURIER_TRANSFORM_KERNEL_H_
#define _DISCRETE_FOURIER_TRANSFORM_KERNEL_H_

#include <kernelized_correlation_filters/cuda_common.h>

cufftComplex* convertFloatToComplexGPU(const float *dev_data,
                                          const int FILTER_BATCH,
                                          const int FILTER_SIZE);

#endif /* _DISCRETE_FOURIER_TRANSFORM_KERNEL_H_ */
