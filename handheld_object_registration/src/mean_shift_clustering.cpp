
#include <handheld_object_registration/mean_shift_clustering.h>

float vectorProduct(
    const std::vector<float> point_a, const std::vector<float> point_b) {
    float d_product = 0.0f;
    for (int i = 0; i < point_a.size() - 1; i++) {
       d_product += (point_a[i] * point_b[i]);
    }
    return d_product;
}

float vectorL2Norm(const std::vector<float> point) {
    float sum = 0.0f;
    for (int i = 0; i < point.size(); i++) {
       sum += (point[i] * point[i]);
    }
    return static_cast<float>(sqrt(sum));
}

float euclidean_distance(
    const std::vector<float> &point_a, const std::vector<float> &point_b) {
    // if (point_a.size() != point_b.size()) {
    //    return FLT_MAX;
    // }
    // float dot_prod = vectorProduct(point_a, point_b);
    // float pa_norm = vectorL2Norm(point_a);
    // float pb_norm = vectorL2Norm(point_b);
    // float tetha = std::acos(dot_prod / (pa_norm * pb_norm));
    
    float total = 0;
    for (int i = 0; i < point_a.size(); i++) {
       total += (point_a[i] - point_b[i]) * (point_a[i] - point_b[i]);
    }
    
    return sqrt(total);
}

float gaussian_kernel(float distance, float kernel_bandwidth) {
    float temp =  exp(-(distance*distance) / (kernel_bandwidth));
    return temp;
}

void MeanShift::set_kernel(float (*_kernel_func)(float, float)) {
    if (!_kernel_func) {
        kernel_func = gaussian_kernel;
    } else {
        kernel_func = _kernel_func;
    }
}

std::vector<float> MeanShift::shift_point(
    const std::vector<float> &point,
    const std::vector<std::vector<float> > &points, float kernel_bandwidth) {
    std::vector<float> shifted_point = point;
    for (int dim = 0; dim < shifted_point.size(); dim++) {
        shifted_point[dim] = 0;
    }
    float total_weight = 0;
    for (int i = 0; i < points.size(); i++) {
        std::vector<float> temp_point = points[i];

        
        
        float distance = euclidean_distance(point, temp_point);


        
        float weight = kernel_func(distance, kernel_bandwidth);
        for (int j = 0; j < shifted_point.size(); j++) {
            shifted_point[j] += temp_point[j] * weight;
        }
        total_weight += weight;
    }

    for (int i = 0; i < shifted_point.size(); i++) {
        shifted_point[i] /= total_weight;
    }
    return shifted_point;
}

std::vector<std::vector<float> > MeanShift::cluster(
    std::vector<std::vector<float> > points, float kernel_bandwidth) {
    std::vector<bool> stop_moving;
    stop_moving.reserve(points.size());
    std::vector<std::vector<float> > shifted_points = points;
    float max_shift_distance;
    do {
        max_shift_distance = 0;
        for (int i = 0; i < shifted_points.size(); i++) {
            if (!stop_moving[i]) {
                std::vector<float> point_new = shift_point(
                   shifted_points[i], points, kernel_bandwidth);
                float shift_distance = euclidean_distance(
                   point_new, shifted_points[i]);
                if (shift_distance > max_shift_distance) {
                   max_shift_distance = shift_distance;
                }
                if (shift_distance <= EPSILON) {
                    stop_moving[i] = true;
                }
                shifted_points[i] = point_new;
            }
        }
        // printf("max_shift_distance: %f\n", max_shift_distance);
        
    } while (max_shift_distance > EPSILON);
    return shifted_points;
}
