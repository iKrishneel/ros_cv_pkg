
#include <depth_map_icp/depth_map_icp.h>

DepthMapICP::DepthMapICP() :
    is_init_(false) {
    this->onInit();
}

void DepthMapICP::onInit() {
    this->subscribe();
    this->pub_depth_ = pnh_.advertise<sensor_msgs::Image>(
       "target", 1);
}

void DepthMapICP::subscribe() {
    this->sub_depth_ = this->pnh_.subscribe(
       "depth", 1, &DepthMapICP::depthCB, this);
}

void DepthMapICP::unsubscribe() {
    this->sub_depth_.shutdown();
}

void DepthMapICP::depthCB(
    const sensor_msgs::Image::ConstPtr &image_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(
          image_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    const float MAX_DIST = 10.0f;
    const float MIN_DIST = 0.0f;
    cv::Mat depth_image;
    cv_ptr->image.convertTo(depth_image, CV_8UC1,
                            255/(MAX_DIST - MIN_DIST), -MIN_DIST);
    
    if (!is_init_) {
       ROS_WARN("SETTING THE INIT FRAME");
       this->prev_depth_ = depth_image;
       is_init_ = true;
       return;
    }
    
    cv::namedWindow("depth", cv::WINDOW_NORMAL);
    cv::imshow("depth", depth_image);
    cv::waitKey(3);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "depth_map_icp");
    DepthMapICP dmi;
    ros::spin();
    return 0;
}


