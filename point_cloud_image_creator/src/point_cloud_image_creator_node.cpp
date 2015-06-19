
#include <point_cloud_image_creator/point_cloud_image_creator.h>
#include <iostream>
#include <string>
#include <vector>

PointCloudImageCreator::PointCloudImageCreator() {
    this->subsribe();
    this->onInit();
}

void PointCloudImageCreator::onInit() {
    this->pub_image_ = pnh_.advertise<sensor_msgs::Image>(
       "/cloud_image/output/image", 1);
    this->pub_iimage_ = pnh_.advertise<sensor_msgs::Image>(
       "/cloud_image/output/interpolated_image", 1);
    this->pub_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>(
       "/cloud_image/output/cloud", 1);
}

void PointCloudImageCreator::subsribe() {
   
    this->sub_cam_info_ = this->pnh_.subscribe(
       "input_info", 1, &PointCloudImageCreator::cameraInfoCallback, this);
   
    this->sub_cloud_ = this->pnh_.subscribe(
       "input", 1, &PointCloudImageCreator::cloudCallback, this);
}

void PointCloudImageCreator::cloudCallback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    boost::mutex::scoped_lock lock(this->lock_);
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    cv::Mat mask;
    cv::Mat img_out = this->projectPointCloudToImagePlane(
       cloud, this->camera_info_, mask);
    cv::Mat interpolate_img = this->interpolateImage(img_out, mask);

    cv_bridge::CvImagePtr image_msg(new cv_bridge::CvImage);
    image_msg->header = cloud_msg->header;
    image_msg->encoding = sensor_msgs::image_encodings::BGR8;
    image_msg->image = img_out.clone();

    cv_bridge::CvImagePtr iimage_msg(new cv_bridge::CvImage);
    iimage_msg->header = cloud_msg->header;
    iimage_msg->encoding = sensor_msgs::image_encodings::BGR8;
    iimage_msg->image = interpolate_img.clone();
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;

    pub_image_.publish(image_msg->toImageMsg());
    pub_iimage_.publish(iimage_msg->toImageMsg());
    pub_cloud_.publish(ros_cloud);
}

void PointCloudImageCreator::cameraInfoCallback(
    const sensor_msgs::CameraInfo::ConstPtr &info_msg) {
    boost::mutex::scoped_lock lock(this->lock_);
    camera_info_ = info_msg;
}

cv::Mat PointCloudImageCreator::projectPointCloudToImagePlane(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const sensor_msgs::CameraInfo::ConstPtr &camera_info,
    cv::Mat &mask) {
    if (cloud->empty()) {
       ROS_ERROR("INPUT CLOUD EMPTY");
       return cv::Mat();
    }
    cv::Mat objectPoints = cv::Mat(static_cast<int>(cloud->size()), 3, CV_32F);
    for (int i = 0; i < cloud->size(); i++) {
       objectPoints.at<float>(i, 0) = cloud->points[i].x;
       objectPoints.at<float>(i, 1) = cloud->points[i].y;
       objectPoints.at<float>(i, 2) = cloud->points[i].z;
    }
    float K[9];
    float R[9];
    for (int i = 0; i < 9; i++) {
       K[i] = camera_info->K[i];
       R[i] = camera_info->R[i];
    }
    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32F, K);
    cv::Mat rotationMatrix = cv::Mat(3, 3, CV_32F, R);
    float tvec[3];
    tvec[0] = camera_info->P[3];
    tvec[1] = camera_info->P[7];
    tvec[2] = camera_info->P[11];
    cv::Mat translationMatrix = cv::Mat(3, 1, CV_32F, tvec);

    float D[5];
    for (int i = 0; i < 5; i++) {
       D[i] = camera_info->D[i];
    }
    cv::Mat distortionModel = cv::Mat(5, 1, CV_32F, D);
    cv::Mat rvec;
    cv::Rodrigues(rotationMatrix, rvec);
    
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(objectPoints, rvec, translationMatrix,
                      cameraMatrix, distortionModel, imagePoints);
    cv::Scalar white = cv::Scalar(255, 255, 255);
    cv::Mat image = cv::Mat(
       camera_info->height, camera_info->width, CV_8UC3, white);
    mask = cv::Mat::zeros(
       camera_info->height, camera_info->width, CV_32F);
    for (int i = 0; i < imagePoints.size(); i++) {
       int x = imagePoints[i].x;
       int y = imagePoints[i].y;
       if (!isnan(x) && !isnan(y) && (x >= 0 && x <= image.cols) &&
           (y >= 0 && y <= image.rows)) {
          image.at<cv::Vec3b>(y, x)[2] = cloud->points[i].r;
          image.at<cv::Vec3b>(y, x)[1] = cloud->points[i].g;
          image.at<cv::Vec3b>(y, x)[0] = cloud->points[i].b;

          mask.at<float>(y, x) = 255.0f;
       }
    }
    // cv::imshow("image", mask);
    return image;
}

cv::Mat PointCloudImageCreator::interpolateImage(
    const cv::Mat &image, const cv::Mat &mask) {
    if (image.empty()) {
       return image;
    }
    cv::Mat iimg = image.clone();
    cv::Mat mop_imgg;
    cv::Mat mop_img;
    this->cvMorphologicalOperations(image, mop_imgg, false);
    this->cvMorphologicalOperations(mop_imgg, mop_img, false);
    
    const int neigbour = 1;
    for (int j = neigbour; j < mask.rows - neigbour; j++) {
       for (int i = neigbour; i < mask.cols - neigbour; i++) {
          if (mask.at<float>(j, i) == 0) {
             int p0 = 0;
             int p1 = 0;
             int p2 = 0;
             int icnt = 0;
             for (int y = -neigbour; y < neigbour + 1; y++) {
                for (int x = -neigbour; x < neigbour + 1; x++) {
                   if (x != i && y != j) {
                      p0 += mop_img.at<cv::Vec3b>(j + y, i + x)[0];
                      p1 += mop_img.at<cv::Vec3b>(j + y, i + x)[1];
                      p2 += mop_img.at<cv::Vec3b>(j + y, i + x)[2];
                      icnt++;
                   }
                }
             }
             iimg.at<cv::Vec3b>(j, i)[0] = p0 / icnt;
             iimg.at<cv::Vec3b>(j, i)[1] = p1 / icnt;
             iimg.at<cv::Vec3b>(j, i)[2] = p2 / icnt;
          }
       }
    }
    return iimg;
}

void PointCloudImageCreator::cvMorphologicalOperations(
    const cv::Mat &img, cv::Mat &erosion_dst, bool is_errode) {
    if (img.empty()) {
       ROS_ERROR("Cannnot perfrom Morphological Operations on empty image....");
       return;
    }
    int erosion_size = 5;
    int erosion_const = 5;
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

    ros::init(argc, argv, "point_cloud_image_creator");
    PointCloudImageCreator pcic;
    ros::spin();
}
