
#include <interactive_segmentation/interactive_segmentation.h>
#include <vector>

InteractiveSegmentation::InteractiveSegmentation() {
    this->subscribe();
    this->onInit();
}

void InteractiveSegmentation::onInit() {
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/interactive_segmentation/output/cloud", 1);
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/interactive_segmentation/output/image", 1);
}

void InteractiveSegmentation::subscribe() {
       this->sub_image_.subscribe(this->pnh_, "input_image", 1);
       this->sub_edge_.subscribe(this->pnh_, "input_edge", 1);
       this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
       this->sync_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy> >(100);
       sync_->connectInput(sub_image_, sub_edge_, sub_cloud_);
       sync_->registerCallback(boost::bind(&InteractiveSegmentation::callback,
                                           this, _1, _2, _3));
}

void InteractiveSegmentation::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_image_.unsubscribe();
}


void InteractiveSegmentation::callback(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const sensor_msgs::Image::ConstPtr &edge_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    boost::mutex::scoped_lock lock(this->mutex_);
    cv::Mat image = cv_bridge::toCvShare(
       image_msg, image_msg->encoding)->image;
    cv::Mat edge_img = cv_bridge::toCvShare(
       edge_msg, edge_msg->encoding)->image;
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    this->pointCloudEdge(image, edge_img, 10);
    
    
    cv_bridge::CvImage pub_img(
       image_msg->header, sensor_msgs::image_encodings::BGR8, image);
    this->pub_image_.publish(pub_img.toImageMsg());

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

void InteractiveSegmentation::pointCloudEdge(
    const cv::Mat &image, const cv::Mat &edge_img, const int contour_thresh) {
    if (image.empty()) {
       ROS_ERROR("-- Cannot eompute edge of empty image");
       return;
    }
    // cv::Mat edge_image;
    // this->getRGBEdge(image, edge_image, "cvCanny");

    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat cont_img = cv::Mat::zeros(image.size(), CV_8U);
    cv::findContours(edge_img, contours, hierarchy, CV_RETR_LIST,
                     CV_CHAIN_APPROX_TC89_KCOS, cv::Point(0, 0));
    std::vector<std::vector<cv::Point> > selected_contours;
    for (int i = 0; i < contours.size(); i++) {
       if (cv::contourArea(contours[i]) > contour_thresh) {
          selected_contours.push_back(contours[i]);
          drawContours(
             cont_img, contours, i, cv::Scalar(0, 255, 0), 1, 8,
             hierarchy, 0, cv::Point());
       }
    }
    imshow("Contours", cont_img);
    imshow("edge", edge_img);
    cv::waitKey(3);
    
}

void InteractiveSegmentation::getRGBEdge(
    const cv::Mat &img, cv::Mat &edgeMap,
    std::string type) {
    if (img.empty()) {
       ROS_ERROR("-- Cannot find edge of empty RGB image");
       return;
    }
    cv::Mat img_;
    cv::cvtColor(img, img_, CV_BGR2GRAY);
    if (type == "cvDOG") {
       cv::Mat dog1;
       cv::Mat dog2;
       cv::GaussianBlur(img_, dog1, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
       cv::GaussianBlur(img_, dog2, cv::Size(21, 21), 0, 0, cv::BORDER_DEFAULT);
       edgeMap = dog1 - dog2;
    } else if (type == "cvSOBEL") {
       int scale = 1;
       int delta = 0;
       int ddepth = CV_8UC1;
       cv::Mat grad_x;
       cv::Mat grad_y;
       cv::Mat abs_grad_x;
       cv::Mat abs_grad_y;
       cv::Sobel(
          img_, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
       cv::convertScaleAbs(grad_x, abs_grad_x);
       cv::Sobel(
          img_, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
       cv::convertScaleAbs(grad_y, abs_grad_y);
       cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edgeMap);
    } else if (type == "cvCanny") {
       cv:Canny(img_, edgeMap, 10, 100, 3);
    }
    cv::Mat edge_mp;
    this->cvMorphologicalOperations(edgeMap, edge_mp, false, 1);
    this->cvMorphologicalOperations(edgeMap, edgeMap, true, 1);
    
    cv::imshow("canny", edge_mp);
}

void InteractiveSegmentation::cvMorphologicalOperations(
    const cv::Mat &img, cv::Mat &erosion_dst,
    bool is_errode, int erosion_size) {
    if (img.empty()) {
       ROS_ERROR("Cannnot perfrom Morphological Operations on empty image....");
       return;
    }
    int erosion_const = 2;
    int erosion_type = cv::MORPH_ELLIPSE;
    cv::Mat element = cv::getStructuringElement(erosion_type,
       cv::Size(erosion_const * erosion_size + sizeof(char),
                erosion_const * erosion_size + sizeof(char)),
       cv::Point(erosion_size, erosion_size));
    if (is_errode) {
       cv::erode(img, erosion_dst, element);
    } else {
       cv::dilate(img, erosion_dst, element);
    }
}



int main(int argc, char *argv[]) {
    ros::init(argc, argv, "interactive_segmentation");
    InteractiveSegmentation is;
    ros::spin();
    return 0;
}
