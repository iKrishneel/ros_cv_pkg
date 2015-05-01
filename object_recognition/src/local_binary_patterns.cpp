
#include <object_recognition/local_binary_patterns.h>

LocalBinaryPatterns::LocalBinaryPatterns() {
   
}

template <typename _Tp>
void LocalBinaryPatterns::localBinaryPatterns(
    const cv::Mat &src, cv::Mat &dst) {
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            _Tp center = src.at<_Tp>(i, j);
            unsigned char code = 0;
            code |= (src.at<_Tp>(i-1, j-1) > center) << 7;
            code |= (src.at<_Tp>(i-1, j) > center) << 6;
            code |= (src.at<_Tp>(i-1, j+1) > center)  << 5;
            code |= (src.at<_Tp>(i, j+1) > center) << 4;
            code |= (src.at<_Tp>(i+1, j+1) > center) << 3;
            code |= (src.at<_Tp>(i+1, j) > center) << 2;
            code |= (src.at<_Tp>(i+1, j-1) > center) << 1;
            code |= (src.at<_Tp>(i, j-1) > center) << 0;
            dst.at<unsigned char>(i, j) = code;
        }
    }
}

void LocalBinaryPatterns::getLBP(
    const cv::Mat& src, cv::Mat& dst) {
    switch (src.type()) {
       case CV_8SC1: localBinaryPatterns<char>(src, dst);
          break;
       case CV_8UC1: localBinaryPatterns<unsigned char>(src, dst);
          break;
       case CV_16SC1: localBinaryPatterns<short>(src, dst);
          break;
       case CV_16UC1: localBinaryPatterns<unsigned short>(src, dst);
          break;
       case CV_32SC1: localBinaryPatterns<int>(src, dst);
          break;
       case CV_32FC1: localBinaryPatterns<float>(src, dst);
          break;
       case CV_64FC1: localBinaryPatterns<double>(src, dst);
          break;
    }
}


cv::Mat LocalBinaryPatterns::histogramLBP(
    const cv::Mat& src, int minVal, int maxVal, bool normed) {
    cv::Mat result;
    int histSize = maxVal - minVal + sizeof(char);
    float range[] = { static_cast<float>(minVal),
                      static_cast<float>(maxVal + sizeof(char))};
    const float* histRange = {range};
    calcHist(
       &src, 1, 0, cv::Mat(), result, 1, &histSize, &histRange, true, false);
    if (normed) {
       normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    }
    return result;
}

void LocalBinaryPatterns::patchWiseLBP(
    const cv::Mat &lbpMD, cv::Mat &lbp_histogram,
    const cv::Size psize, int bin_sz, bool isnorm) {
    for (int j = 0; j < lbpMD.rows; j += psize.height) {
       for (int i = 0; i < lbpMD.cols; i += psize.width) {
          cv::Rect_<int> rect = cv::Rect_<int>(i, j, psize.width, psize.height);
          if ((rect.x + rect.width < lbpMD.cols) &&
              (rect.y + rect.height < lbpMD.rows)) {
             cv::Mat roi = lbpMD(rect).clone();
             lbp_histogram.push_back(
                this->histogramLBP(roi, 0, bin_sz - sizeof(char), isnorm));
          }
       }
    }
    lbp_histogram = lbp_histogram.reshape(1, 1);
}


cv::Mat LocalBinaryPatterns::computeLBP(
    const cv::Mat &image, const cv::Size psize, const int bin_sz,
    bool full_lbp, bool isnorm) {
    if (image.empty()) {
       ROS_ERROR("-- EMPTY IMAGE IN LBP");
       return cv::Mat();
    }
    cv::Mat img = image.clone();
    if (image.type() != CV_8UC1) {
       cv::cvtColor(img, img, CV_BGR2GRAY);
    }
    cv::Mat lbpMD;
    // this->localBinaryPatterns<uchar>(img, lbpMD);
    this->getLBP(img, lbpMD);
    if (full_lbp) {
       return lbpMD;
    } else {
       cv::Mat lbp_histogram;
       this->patchWiseLBP(lbpMD, lbp_histogram, psize, bin_sz, isnorm);
       return lbp_histogram;
    }
}
