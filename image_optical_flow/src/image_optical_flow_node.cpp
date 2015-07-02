// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo

#include <image_optical_flow/image_optical_flow.h>
#include <vector>

ImageOpticalFlow::ImageOpticalFlow() :
    icounter(0) {
    this->detector_ = cv::FeatureDetector::create("FAST");
    this->descriptor_ = cv::DescriptorExtractor::create("ORB");
    
    this->prev_pts_.clear();
    this->subscribe();
    this->onInit();
}

void ImageOpticalFlow::onInit() {
    this->pub_image_ = pnh_.advertise<sensor_msgs::Image>(
       "target", 1);
   
}

void ImageOpticalFlow::subscribe() {
    this->sub_image_ = this->pnh_.subscribe(
       "input", 1, &ImageOpticalFlow::imageCallback, this);
}

void ImageOpticalFlow::unsubscribe() {
    this->sub_image_.shutdown();
}

void ImageOpticalFlow::imageCallback(
    const sensor_msgs::Image::ConstPtr &image_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(
           image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat image = cv_ptr->image.clone();
    cv::Mat img;
    cv::cvtColor(image, img, CV_BGR2GRAY);
    cv::goodFeaturesToTrack(
       img, this->prev_pts_, 300, 0.01, 10, cv::Mat(), 3, false, 0.04);
    if (icounter++ != 0) {
       FeatureInfo info;
       forwardBackwardMatchingAndFeatureCorrespondance(
          this->prev_image_, image, info);
    }
    this->prev_image_ = cv_ptr->image.clone();
    
    ros::Duration(5).sleep();
    
    cv_bridge::CvImagePtr pub_msg(new cv_bridge::CvImage);
    pub_msg->header = image_msg->header;
    pub_msg->encoding = sensor_msgs::image_encodings::BGR8;
    pub_msg->image = image.clone();
    this->pub_image_.publish(pub_msg);
}

void ImageOpticalFlow::buildImagePyramid(
    const cv::Mat &frame,
    std::vector<cv::Mat> &pyramid) {
    cv::Mat gray = frame.clone();
    cv::Size winSize = cv::Size(5, 5);
    int maxLevel = 5;
    bool withDerivative = true;
    cv::buildOpticalFlowPyramid(
       gray, pyramid, winSize, maxLevel, withDerivative,
       cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, true);
    
}

void ImageOpticalFlow::getOpticalFlow(
    const cv::Mat &frame, const cv::Mat &prevFrame,
    std::vector<cv::Point2f> &nextPts, std::vector<cv::Point2f> &prevPts,
    std::vector<uchar> &status) {
    cv::Mat gray, grayPrev;
    cv::cvtColor(prevFrame, grayPrev, CV_BGR2GRAY);
    cv::cvtColor(frame, gray, CV_BGR2GRAY);
    std::vector<cv::Mat> curPyramid;
    std::vector<cv::Mat> prevPyramid;
    buildImagePyramid(frame, curPyramid);
    buildImagePyramid(prevFrame, prevPyramid);
    std::vector<float> err;
    nextPts.clear();
    status.clear();
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
    cv::Mat iFrame = prevFrame.clone();
}


void ImageOpticalFlow::forwardBackwardMatchingAndFeatureCorrespondance(
    const cv::Mat img1, const cv::Mat img2, FeatureInfo &info) {
    std::vector<cv::Point2f> nextPts;
    std::vector<cv::Point2f> prevPts;
    std::vector<cv::Point2f> backPts;
    cv::GaussianBlur(img1, img1, cv::Size(5, 5), 1);
    cv::GaussianBlur(img2, img2, cv::Size(5, 5), 1);
    cv::Mat gray;
    cv::Mat grayPrev;
    cv::cvtColor(img1, grayPrev, CV_BGR2GRAY);
    cv::cvtColor(img2, gray, CV_BGR2GRAY);
    std::vector<cv::KeyPoint> keypoints_prev;
    this->detector_->detect(grayPrev, keypoints_prev);
    for (int i = 0; i < keypoints_prev.size(); i++) {
       prevPts.push_back(keypoints_prev[i].pt);
    }
    std::vector<uchar> status;
    std::vector<uchar> status_back;
    this->getOpticalFlow(img2, img1, nextPts, prevPts, status);
    this->getOpticalFlow(img1, img2, backPts, nextPts, status_back);
    std::vector<float> fb_err;
    for (int i = 0; i < prevPts.size(); i++) {
       cv::Point2f v = backPts[i] - prevPts[i];
       fb_err.push_back(sqrt(v.dot(v)));
    }
    float THESHOLD = 10;
    for (int i = 0; i < status.size(); i++) {
       status[i] = (fb_err[i] <= THESHOLD) & status[i];
    }
    std::vector<cv::KeyPoint> keypoints_next;
    for (int i = 0; i < prevPts.size(); i++) {
       cv::Point2f ppt = prevPts[i];
       cv::Point2f npt = nextPts[i];
       double distance = cv::norm(cv::Mat(ppt), cv::Mat(npt));
       if (status[i] && distance > 5) {
          cv::KeyPoint kp;
          kp.pt = nextPts[i];
          kp.size = keypoints_prev[i].size;
          keypoints_next.push_back(kp);
       }
    }
    std::vector<cv::KeyPoint>keypoints_cur;
    this->detector_->detect(img2, keypoints_cur);
    std::vector<cv::KeyPoint> keypoints_around_region;
    for (int i = 0; i < keypoints_cur.size(); i++) {
       cv::Point2f cur_pt = keypoints_cur[i].pt;
       for (int j = 0; j < keypoints_next.size(); j++) {
          cv::Point2f est_pt = keypoints_next[j].pt;
          double distance = cv::norm(cv::Mat(cur_pt), cv::Mat(est_pt));
          if (distance < 10) {
             keypoints_around_region.push_back(keypoints_cur[i]);
          }
       }
    }
    cv::Mat descriptor_cur;
    this->descriptor_->compute(img2, keypoints_around_region, descriptor_cur);
    cv::Mat descriptor_prev;
    this->descriptor_->compute(img1, keypoints_prev, descriptor_prev);
    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher =
       cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<std::vector<cv::DMatch> > matchesAll;
    descriptorMatcher->knnMatch(descriptor_cur, descriptor_prev, matchesAll, 2);
    std::vector<cv::DMatch> match1;
    std::vector<cv::DMatch> match2;
    for (int i=0; i < matchesAll.size(); i++) {
       match1.push_back(matchesAll[i][0]);
       match2.push_back(matchesAll[i][1]);
    }
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < matchesAll.size(); i++) {
       if (match1[i].distance < 0.5 * match2[i].distance) {
          good_matches.push_back(match1[i]);
       }
    }
    cv::Mat img_matches1;
    drawMatches(img2, keypoints_around_region, img1,
                keypoints_prev, good_matches, img_matches1);
    cv::imshow("matches1", img_matches1);
    cv::waitKey(0);
}



int main(int argc, char *argv[]) {
    ros::init(argc, argv, "image_optical_flow");
    ImageOpticalFlow iof;
    ros::spin();
}
