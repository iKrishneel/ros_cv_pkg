
#pragma once
#ifndef _FAST_MATHS_KERNEL_H_
#define _FAST_MATHS_KERNEL_H_

#include <kernelized_correlation_filters/cuda_common.h>


cufftComplex* multiplyComplexGPU(const cufftComplex *,
                                 const cufftComplex *,
                                 const int);

cufftComplex* multiplyComplexByScalarGPU(const cufftComplex *,
                                         const float,
                                         const int);

cufftComplex* addComplexGPU(const cufftComplex *,
                            const cufftComplex *,
                            const int);

cufftComplex* divisionComplexGPU(const cufftComplex *,
                                 const cufftComplex *,
                                 const int);

#endif /* _FAST_MATHS_KERNEL_H_ */
