
#ifndef _BILINEAR_INTERPOLATION_KERNEL_H_
#define _BILINEAR_INTERPOLATION_KERNEL_H_

#include <kernelized_correlation_filters/cuda_common.h>

float *bilinearInterpolationGPU(const float *, const int, const int,
                                const int, const int, const int, const int);

float *bilinear_test(float *data, const int in_byte);

#endif /* _BILINEAR_INTERPOLATION_KERNEL_H_ */
