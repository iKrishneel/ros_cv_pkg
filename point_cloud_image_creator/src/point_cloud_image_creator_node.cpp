
#include <point_cloud_image_creator/point_cloud_image_creator.h>
#include <iostream>
#include <string>
#include <vector>

PointCloudImageCreator::PointCloudImageCreator() {

    this->subsribe();
    this->onInit();
}

void PointCloudImageCreator::onInit() {
    pub_image_ = pnh_.advertise<sensor_msgs::Image>("output/image", 1);
    pub_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>("output/cloud", 1);
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

    
    cv_bridge::CvImagePtr image_msg(new cv_bridge::CvImage);
    image_msg->header = cloud_msg->header;
    image_msg->encoding = sensor_msgs::image_encodings::BGR8;
    image_msg->image = this->projectPointCloudToImagePlane(
       cloud, this->camera_info_).clone();
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;

    pub_image_.publish(image_msg->toImageMsg());
    pub_cloud_.publish(ros_cloud);
}

void PointCloudImageCreator::cameraInfoCallback(
    const sensor_msgs::CameraInfo::ConstPtr &info_msg) {
    boost::mutex::scoped_lock lock(this->lock_);
    camera_info_ = info_msg;
}

cv::Mat PointCloudImageCreator::projectPointCloudToImagePlane(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const sensor_msgs::CameraInfo::ConstPtr &camera_info) {
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
    cv::Mat image = cv::Mat::zeros(
       camera_info->height, camera_info->width, CV_8UC3);
    for (int i = 0; i < imagePoints.size(); i++) {
       int x = imagePoints[i].x;
       int y = imagePoints[i].y;
       if (!isnan(x) && !isnan(y) && (x >= 0 && x <= image.cols) &&
           (y >= 0 && y <= image.rows)) {
          image.at<cv::Vec3b>(y, x)[2] = cloud->points[i].r;
          image.at<cv::Vec3b>(y, x)[1] = cloud->points[i].g;
          image.at<cv::Vec3b>(y, x)[0] = cloud->points[i].b;
       }
    }
    // cv::imshow("image", image);
    // cv::waitKey(3);
    return image;
}


int main(int argc, char *argv[]) {

    ros::init(argc, argv, "point_cloud_image_creator");
    PointCloudImageCreator pcic;
    ros::spin();
}
