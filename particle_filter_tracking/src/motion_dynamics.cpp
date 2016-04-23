
//  Created by Chaudhary Krishneel on 3/28/14.
//  Copyright (c) 2014 Chaudhary Krishneel. All rights reserved.


#include <particle_filter_tracking/motion_dynamics.h>

MotionDynamics::MotionDynamics() : feature_size(300) {
   
}

void MotionDynamics::buildImagePyramid(
    const cv::Mat &frame, std::vector<cv::Mat> &pyramid) {
    cv::Mat gray = frame.clone();
    cv::Size winSize = cv::Size(5, 5);
    int maxLevel = 3;
    bool withDerivative = true;
    cv::buildOpticalFlowPyramid(
       gray, pyramid, winSize, maxLevel, withDerivative,
       cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, true);
    
}

void MotionDynamics::getOpticalFlow(
    const cv::Mat &frame, const cv::Mat &prevFrame,
    std::vector<cv::Point2f> &prevPts, bool doPyramid) {
    if (doPyramid) {
        cv::Mat gray, grayPrev;
        cv::cvtColor(prevFrame, grayPrev, CV_BGR2GRAY);
        cv::cvtColor(frame, gray, CV_BGR2GRAY);
        std::vector<cv::Mat> curPyramid;
        std::vector<cv::Mat> prevPyramid;
        this->buildImagePyramid(frame, curPyramid);
        this->buildImagePyramid(prevFrame, prevPyramid);
        std::vector<cv::Point2f> nextPts;
        cv::goodFeaturesToTrack(gray, nextPts, feature_size, 0.01, 0.1);
        cv::goodFeaturesToTrack(grayPrev, prevPts, feature_size, 0.01, 0.1);
        std::vector<uchar> status;
        std::vector<float> err;
        nextPts.resize(prevPts.size());
        status.resize(prevPts.size());
        err.resize(prevPts.size());
        cv::Size winSize = cv::Size(5, 5);
        int maxLevel = 3;
        cv::TermCriteria criteria = cv::TermCriteria(
           cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
        int flags = 0;
        cv::calcOpticalFlowPyrLK(
           prevPyramid, curPyramid, prevPts, nextPts, status,
           err, winSize, maxLevel, criteria, flags);
    } else {
       cv::Mat gray;
        cv::Mat grayPrev;
        cv::cvtColor(prevFrame, grayPrev, CV_BGR2GRAY);
        cv::cvtColor(frame, gray, CV_BGR2GRAY);
        std::vector<cv::Point2f> nextPts;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::Size winSize = cv::Size(16, 16);
        int maxLevel = 0;
        cv::TermCriteria criteria = cv::TermCriteria(
           cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
        cv::calcOpticalFlowPyrLK(
           grayPrev, gray, prevPts, nextPts, status,
           err, winSize, maxLevel, criteria, cv::OPTFLOW_LK_GET_MIN_EIGENVALS);
        prevPts.clear();
        prevPts = nextPts;
        nextPts.clear();
    }
}
