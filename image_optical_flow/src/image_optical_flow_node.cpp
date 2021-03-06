// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo

#include <image_optical_flow/image_optical_flow.h>
#include <vector>

ImageOpticalFlow::ImageOpticalFlow() :
    icounter(0) {
    this->prev_pts_.clear();
    // this->subscribe();
    // this->onInit();

    cv::Mat img1 = cv::imread("/home/krishneel/Desktop/frame0000.jpg", 0);
    cv::Mat img2 = cv::imread("/home/krishneel/Desktop/frame0001.jpg", 0);
    cv::goodFeaturesToTrack(
       img1, this->prev_pts_, 1000, 0.01, 10, cv::Mat(), 3, false, 0.04);
    std::vector<cv::Point2f> next_pts;
    cv::goodFeaturesToTrack(
       img2, next_pts, 1000, 0.01, 10, cv::Mat(), 3, false, 0.04);

    cv::cvtColor(img1, img1, CV_GRAY2BGR);
    cv::cvtColor(img2, img2, CV_GRAY2BGR);
    this->computeOpticalFlow(
       img1, img2, this->prev_pts_, next_pts);
    this->drawFlowField(img2, this->prev_image_, next_pts, this->prev_pts_);

    cv::imshow("flow", img2);
    cv::waitKey(0);
    
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
       img, this->prev_pts_, 500, 0.01, 10, cv::Mat(), 3, false, 0.04);
    if (icounter++ != 0) {
       std::vector<cv::Point2f> next_pts;
       this->computeOpticalFlow(
          this->prev_image_, image, this->prev_pts_, next_pts);
       this->drawFlowField(image, this->prev_image_, next_pts, this->prev_pts_);
    }
    this->prev_image_ = cv_ptr->image.clone();
    cv_bridge::CvImagePtr pub_msg(new cv_bridge::CvImage);
    pub_msg->header = image_msg->header;
    pub_msg->encoding = sensor_msgs::image_encodings::BGR8;
    pub_msg->image = image.clone();
    this->pub_image_.publish(pub_msg);
}

void ImageOpticalFlow::computeOpticalFlow(
    cv::Mat &prev_frame, cv::Mat &cur_frame,
    std::vector<cv::Point2f> &prev_pts,
    std::vector<cv::Point2f> &next_pts) {
    if (prev_frame.empty() || cur_frame.empty()) {
       ROS_ERROR("-- Cannot compute flow of empty image");
       return;
    }
    next_pts.clear();
    std::vector<uchar> status;
    std::vector<float> err;
    cv::Size win_size = cv::Size(3, 3);
    int max_level = 0;
    cv::TermCriteria criteria = cv::TermCriteria(
       cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    cv::Mat prev_gray;
    cv::Mat cur_gray;
    cv::cvtColor(prev_frame, prev_gray, CV_BGR2GRAY);
    cv::cvtColor(cur_frame, cur_gray, CV_BGR2GRAY);
    cv::calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_pts, next_pts,
                             status, err, win_size, max_level, criteria,
                             cv::OPTFLOW_LK_GET_MIN_EIGENVALS);
}

void ImageOpticalFlow::drawFlowField(
    cv::Mat &frame, cv::Mat &prev_frame,
    std::vector<cv::Point2f> &next_pts,
    std::vector<cv::Point2f> &prev_pts) {
    if (frame.empty()) {
       ROS_ERROR("-- cannot draw flow on empty image");
       return;
    }
    for (int i = 0 ; i < next_pts.size(); i++) {
       double angle = atan2(static_cast<double>(prev_pts[i].y - next_pts[i].y),
                            static_cast<double>(prev_pts[i].x - next_pts[i].x));
       double lenght = sqrt(static_cast<double>(
                               std::pow(prev_pts[i].y - next_pts[i].y, 2)) +
                            static_cast<double>(
                               std::pow(prev_pts[i].x - next_pts[i].x, 2)));
       cv::line(frame, prev_pts[i], next_pts[i], cv::Scalar(0, 255, 0), 2);
       cv::line(prev_frame, prev_pts[i], next_pts[i], cv::Scalar(0, 255, 0), 2);
       cv::Point ipos = cv::Point(prev_pts[i].x - 3 * lenght * std::cos(angle),
                              prev_pts[i].y - 3 * lenght * std::sin(angle));
       cv::line(frame, prev_pts[i], ipos, cv::Scalar(255, 0, 255));
       cv::line(frame, cv::Point2f(ipos.x + 9 * std::cos(angle + CV_PI/4),
                                   ipos.y + 9 * std::sin(angle + CV_PI/4)),
                ipos, cv::Scalar(255, 0, 255));
       cv::line(frame, cv::Point2f(ipos.x + 9 * std::cos(angle - CV_PI/4),
                                   ipos.y + 9 * std::sin(angle - CV_PI/4)),
                ipos, cv::Scalar(255, 0, 255));
    }
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "image_optical_flow");
    ImageOpticalFlow iof;
    ros::spin();
}
