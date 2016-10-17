
#pragma once
#ifndef _GAUSSIAN_CORRELATION_KERNEL_H_
#define _GAUSSIAN_CORRELATION_KERNEL_H_

#include <kernelized_correlation_filters/cuda_common.h>
#include <kernelized_correlation_filters/threadFenceReduction_kernel.h>

float squaredNormGPU(const cufftComplex *, const int, const int);
/* returns both squared norm and mag in single call */
float* squaredNormAndMagGPU(float &, const cufftComplex *,
                            const int, const int);
/* reverse the conjuate*/
cufftComplex* invComplexConjuateGPU(const cufftComplex *,
                                    const int, const int);
/* sums the in complex with reverse conjuate in single call*/
cufftComplex* invConjuateConvGPU(const cufftComplex *,
                                 const cufftComplex *,
                                 const int, const int);
/* sum over filters */
float* invFFTSumOverFiltersGPU(const float *,
                               const int, const int);

#endif /* _GAUSSIAN_CORRELATION_KERNEL_H_ */
