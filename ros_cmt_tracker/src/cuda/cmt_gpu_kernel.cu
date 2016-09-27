
#include <ros_cmt_tracker/cmt_gpu_kernel.h>

struct PairComparator {
    __host__ __device__
    bool operator()(const thrust::pair<float, int> &l,
                            const thrust::pair<float, int> &r) {
       return l.first < r.first;
    }
};


void sortConfPairs(int &index1, int &index2,
                   const std::vector<float> combined) {
    const size_t N = static_cast<int>(combined.size());
    thrust::device_vector<thrust::pair<float, int> > sorted_conf(N);
    
    for (int i = 0; i < combined.size(); i++) {
       sorted_conf[i] = thrust::make_pair(combined[i], i);
    }
    thrust::sort(&sorted_conf[0], &sorted_conf[0] + N,
                 PairComparator());
    // thrust::sort(sorted_conf.begin(), sorted_conf.end(),
    //              PairComparator());
    
    // thrust::pair<float, int> value = sorted_conf[0];
    // index1 = value.second;
    // value = sorted_conf[1];
    // index2 = value.second;
}


void processFrameGPU(cv::Mat im_gray) {
    if (im_gray.empty()) {
       printf("\033[31m EMPTY IMAGE FOR TRACKING \033[0m]\n");
       return;
    }

    size_t N = 16;
    thrust::device_vector<thrust::pair<int, int> > keys(N);
    
}
