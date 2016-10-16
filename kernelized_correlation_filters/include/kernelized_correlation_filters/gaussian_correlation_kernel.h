
#pragma once
#ifndef _GAUSSIAN_CORRELATION_KERNEL_H_
#define _GAUSSIAN_CORRELATION_KERNEL_H_

#include <kernelized_correlation_filters/cuda_common.h>
#include <kernelized_correlation_filters/threadFenceReduction_kernel.h>

float squaredNormGPU(const cufftComplex *, const int, const int);

#endif /* _GAUSSIAN_CORRELATION_KERNEL_H_ */
