
#pragma once
#ifndef _COSINE_CONVOLUTION_KERNEL_H_
#define _COSINE_CONVOLUTION_KERNEL_H_

#include <kernelized_correlation_filters/cuda_common.h>

/*
__host__ __device__
struct caffeFilterInfo {
    int width;
    int height;
    int channels;
    int data_lenght;  //! total lenght = blob->count()
    caffeFilterInfo(int w = -1, int h = -1,
                    int c = -1, int  l = 0) :
       width(w), height(h), channels(c), data_lenght(l) {}
};
*/

float* cosineConvolutionGPU(const float *, const float *,
                          const int, const int);

/*
float *bilinearInterpolationGPU(const float *, const int, const int,
                                const int, const int, const int, const int);

float *bilinear_test(float *data, const int in_byte);
*/

#endif // _COSINE_CONVOLUTION_KERNEL_H_

