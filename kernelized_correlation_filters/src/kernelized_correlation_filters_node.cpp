
#include <kernelized_correlation_filters/kernelized_correlation_filters.h>


KernelizedCorrelationFilters::KernelizedCorrelationFilters() :
    block_size_(8), downsize_(1), tracker_init_(false), threads_(8) {
    this->tracker_ = boost::shared_ptr<KCF_Tracker>(new KCF_Tracker);
    this->onInit();
}

void KernelizedCorrelationFilters::onInit() {
     this->subscribe();
     this->pub_image_ = pnh_.advertise<sensor_msgs::Image>(
         "target", 1);
}

void KernelizedCorrelationFilters::subscribe() {
     this->sub_screen_pt_ = this->pnh_.subscribe(
         "input_screen", 1, &KernelizedCorrelationFilters::screenPtCB, this);
     this->sub_image_ = this->pnh_.subscribe(
         "image", 1, &KernelizedCorrelationFilters::imageCB, this);
}

void KernelizedCorrelationFilters::unsubscribe() {
     this->sub_image_.shutdown();
}

void KernelizedCorrelationFilters::screenPtCB(
     const geometry_msgs::PolygonStamped &screen_msg) {
     int x = screen_msg.polygon.points[0].x;
     int y = screen_msg.polygon.points[0].y;
     int width = screen_msg.polygon.points[1].x - x;
     int height = screen_msg.polygon.points[1].y - y;
     this->screen_rect_ = cv::Rect_<int>(
         x/downsize_, y/downsize_, width/downsize_, height/downsize_);
     if (width > this->block_size_ && height > this->block_size_) {
         this->tracker_init_ = true;
     } else {
         ROS_WARN("-- Selected Object Size is too small... Not init tracker");
     }
}

void KernelizedCorrelationFilters::imageCB(
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
     if (image.empty()) {
         ROS_ERROR("EMPTY INPUT IMAGE");
         return;
     }
     if (downsize_ > 1) {
         cv::resize(image, image, cv::Size(image.cols/this->downsize_,
                                           image.rows/this->downsize_));
     }

     if (this->tracker_init_) {
        ROS_INFO("Initializing Tracker");
        this->tracker_->init(image, this->screen_rect_);
        this->tracker_init_ = false;
        this->prev_frame_ = image(screen_rect_).clone();
        ROS_INFO("Tracker Initialization Complete");
     }

     if (this->screen_rect_.width > this->block_size_) {
        this->tracker_->track(image);
        BBox_c bb = this->tracker_->getBBox();
        cv::Rect rect = cv::Rect(bb.cx - bb.w/2.0f,
                                 bb.cy - bb.h/2.0f, bb.w, bb.h);
        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
        
        
     } else {
        ROS_ERROR_ONCE("THE TRACKER IS NOT INITALIZED");
     }

     cv_bridge::CvImagePtr pub_msg(new cv_bridge::CvImage);
     pub_msg->header = image_msg->header;
     pub_msg->encoding = sensor_msgs::image_encodings::BGR8;
     pub_msg->image = image.clone();
     this->pub_image_.publish(pub_msg);

     cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
     cv::imshow("image", image);
     cv::waitKey(3);
}


int main(int argc, char *argv[]) {
   
    ros::init(argc, argv, "kernelized_correlation_filters");
    KernelizedCorrelationFilters kcf;
    ros::spin();
    return 0;
}

