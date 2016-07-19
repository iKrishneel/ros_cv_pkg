

#include <math.h>
#include <iostream>
#include <vector>

class MeanShift {
   
#define EPSILON 0.001
   
 private:
    float (*kernel_func)(float, float);
    void set_kernel(float (*_kernel_func)(float, float));
    std::vector<float> shift_point(const std::vector<float> &,
                                    const std::vector<std::vector<float> > &,
                                    float);
   
 public:
    MeanShift() {
       set_kernel(NULL);
    }
    MeanShift(float (*_kernel_func)(float, float)) {
       set_kernel(kernel_func); }
    std::vector<std::vector<float> > cluster(std::vector<std::vector<float> >,
                                              float);
};
