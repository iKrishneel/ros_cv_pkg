
#pragma once
#ifndef _DISCRETE_FOURIER_TRANSFORM_KERNEL_H_
#define _DISCRETE_FOURIER_TRANSFORM_KERNEL_H_

#include <kernelized_correlation_filters/cuda_common.h>

cufftComplex* convertFloatToComplexGPU(const float *,
                                       const int, const int);

float* copyComplexRealToFloatGPU(const cufftComplex*,
                                 const int, const int);

void normalizeByFactorGPU(float *&, const float,
                          const int, const int);

#endif /* _DISCRETE_FOURIER_TRANSFORM_KERNEL_H_ */
