
#include <particle_filter_tracking/particle_filter_tracking.h>

ParticleFilterTracking::ParticleFilterTracking() {
    this->subscribe();
    this->onInit();
}

void ParticleFilterTracking::onInit() {
    this->pub_image_ = pnh_.advertise<sensor_msgs::Image>(
       "target", 1);
}

void ParticleFilterTracking::subscribe() {
    this->sub_image_ = this->pnh_.subscribe(
       "input", 1, &ParticleFilterTracking::imageCallback, this);
}

void ParticleFilterTracking::unsubscribe() {
    this->sub_image_.shutdown();
}

void ParticleFilterTracking::imageCallback(
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
    cv_bridge::CvImagePtr pub_msg(new cv_bridge::CvImage);
    pub_msg->header = image_msg->header;
    pub_msg->encoding = sensor_msgs::image_encodings::BGR8;
    pub_msg->image = image.clone();
    this->pub_image_.publish(pub_msg);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "particle_filter_tracking");
    ParticleFilterTracking pft;
    ros::spin();
    return 0;
}
