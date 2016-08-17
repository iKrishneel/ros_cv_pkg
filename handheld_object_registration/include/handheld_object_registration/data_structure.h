
#pragma once
#ifndef _DATA_STRUCTURE_H_
#define _DATA_STRUCTURE_H_

#include <opencv2/opencv.hpp>

struct CandidateIndices {
    int source_index;
    int target_index;
};

struct ProjectionMap {
    int x;
    int y;
    int height;
    int width;
    cv::Mat rgb;
    cv::Mat indices;
    cv::Mat depth;
    std::vector<bool> visibility_flag;
   
    ProjectionMap(int i = -1, int j = -1, int w = -1, int h = -1,
                  cv::Mat image = cv::Mat(),
                  cv::Mat indices_im = cv::Mat(),
                  cv::Mat depth_im = cv::Mat()) :
       x(i), y(j), width(w), height(h), rgb(image),
       indices(indices_im), depth(depth_im) {}
};

#endif /* _DATA_STRUCTURE_H_ */
