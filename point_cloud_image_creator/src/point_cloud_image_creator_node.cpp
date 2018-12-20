
#include <point_cloud_image_creator/point_cloud_image_creator.h>

PointCloudImageCreator::PointCloudImageCreator() {
    pnh_.getParam("mask_image", this->is_mask_image_);
    pnh_.getParam("roi_image", this->is_roi_);
    is_mask_image_ = false;
    is_info_ = false;

    this->pnh_ = ros::NodeHandle("~");
    
    this->subsribe();
    this->onInit();
}

void PointCloudImageCreator::onInit() {

    if (is_mask_image_) {
       this->pub_image_ = pnh_.advertise<sensor_msgs::Image>(
          "/cloud_image/output/image", 1);
       this->pub_fmask_ = pnh_.advertise<sensor_msgs::Image>(
          "/cloud_image/output/foreground_mask", 1);
       this->pub_bmask_ = pnh_.advertise<sensor_msgs::Image>(
          "/cloud_image/output/background_mask", 1);
    } else {
       this->pub_image_ = pnh_.advertise<sensor_msgs::Image>(
          "out_image", 1);
       this->pub_iimage_ = pnh_.advertise<sensor_msgs::Image>(
          "interpolated_image", 1);
       this->pub_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>(
          "out_cloud", 1);
    }
}

void PointCloudImageCreator::subsribe() {

   
    this->sub_cam_info_ = this->pnh_.subscribe(
      "info", 1, &PointCloudImageCreator::cameraInfoCallback, this);
    
    /*
      this->sub_cloud_ = this->pnh_.subscribe(
      "input", 1, &PointCloudImageCreator::cloudCallback, this);

      if (is_mask_image_) {
      this->sub_image_ = this->pnh_.subscribe(
      "in_image", 1, &PointCloudImageCreator::imageCallback, this);
      }
    */

    ROS_INFO("SUBSCRIBING");
   
    this->msub_points_.subscribe(this->pnh_, "points", 10);
    this->msub_indices_.subscribe(this->pnh_, "indices", 10);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(this->msub_points_, this->msub_indices_);
    this->sync_->registerCallback(
       boost::bind(&PointCloudImageCreator::callback, this, _1, _2));
}

void PointCloudImageCreator::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const jsk_recognition_msgs::ClusterPointIndices::ConstPtr &indices_msg) {

    if (!is_info_) {
       ROS_WARN("CAMERA INFO NOT SET");
       return;
    }
    ROS_INFO("PROCESSING");
   
    boost::mutex::scoped_lock lock(this->lock_);
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty()) {
       ROS_WARN("EMPTY CLOUD POINT");
       return;
    }

    cv::Mat mask;
    cv::Mat img_out = this->projectPointCloudToImagePlane(
       cloud, indices_msg, this->camera_info_, mask);

    cv::imshow("mask", img_out);
    cv::waitKey(3);

}

void PointCloudImageCreator::cloudCallback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    if (!is_info_) {
       return;
    }
    boost::mutex::scoped_lock lock(this->lock_);
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty()) {
       return;
    }
    cv::Mat mask;
    cv::Mat img_out = this->projectPointCloudToImagePlane(
       cloud, this->camera_info_, mask);
    cv::Mat interpolate_img = this->interpolateImage(img_out, mask);
    if (is_mask_image_) {
       cv::Mat foreground_mask;
       cv::Mat background_mask;
       this->rect_ = this->createMaskImages(img_out,
                                            foreground_mask,
                                            background_mask);
       this->foreground_mask_ = foreground_mask.clone();
       
       cv_bridge::CvImagePtr fmask_msg(new cv_bridge::CvImage);
       cv_bridge::CvImagePtr bmask_msg(new cv_bridge::CvImage);
       fmask_msg->header = cloud_msg->header;
       fmask_msg->encoding = sensor_msgs::image_encodings::MONO8;
       fmask_msg->image = foreground_mask.clone();
       
       bmask_msg->header = cloud_msg->header;
       bmask_msg->encoding = sensor_msgs::image_encodings::MONO8;
       bmask_msg->image = background_mask.clone();

       this->pub_fmask_.publish(fmask_msg->toImageMsg());
       this->pub_bmask_.publish(bmask_msg->toImageMsg());
    } else {
       sensor_msgs::PointCloud2 ros_cloud;
       pcl::toROSMsg(*cloud, ros_cloud);
       ros_cloud.header = cloud_msg->header;
       pub_cloud_.publish(ros_cloud);

       cv_bridge::CvImagePtr iimage_msg(new cv_bridge::CvImage);
       iimage_msg->header = cloud_msg->header;
       iimage_msg->encoding = sensor_msgs::image_encodings::BGR8;
       iimage_msg->image = interpolate_img.clone();

       cv_bridge::CvImagePtr image_msg(new cv_bridge::CvImage);
       image_msg->header = cloud_msg->header;
       image_msg->encoding = sensor_msgs::image_encodings::BGR8;
       image_msg->image = img_out.clone();
       pub_image_.publish(image_msg->toImageMsg());
       pub_iimage_.publish(iimage_msg->toImageMsg());
    }
    this->header_ = cloud_msg->header;
}

void PointCloudImageCreator::cameraInfoCallback(
    const sensor_msgs::CameraInfo::ConstPtr &info_msg) {
    boost::mutex::scoped_lock lock(this->lock_);
    camera_info_ = info_msg;
    is_info_ = true;
}

void PointCloudImageCreator::imageCallback(
    const sensor_msgs::Image::ConstPtr &image_msg) {
    cv::Mat in_image = cv_bridge::toCvShare(
       image_msg, sensor_msgs::image_encodings::BGR8)->image;
    boost::mutex::scoped_lock lock(mutex_);
    if (is_mask_image_ && !this->foreground_mask_.empty()) {
       in_image = in_image(this->rect_);
    }
    cv_bridge::CvImagePtr out_msg(new cv_bridge::CvImage);
    out_msg->header = this->header_;
    out_msg->encoding = sensor_msgs::image_encodings::BGR8;
    out_msg->image = in_image.clone();
    pub_image_.publish(out_msg->toImageMsg());
}

cv::Mat PointCloudImageCreator::projectPointCloudToImagePlane(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const jsk_recognition_msgs::ClusterPointIndices::ConstPtr indices,
    const sensor_msgs::CameraInfo::ConstPtr &camera_info,
    cv::Mat &mask) {
    if (cloud->empty()) {
       ROS_ERROR("INPUT CLOUD EMPTY");
       return cv::Mat();
    }
    cv::Mat objectPoints = cv::Mat(static_cast<int>(cloud->size()), 3, CV_32F);
    /*
    for (int i = 0; i < cloud->size(); i++) {
       objectPoints.at<float>(i, 0) = cloud->points[i].x;
       objectPoints.at<float>(i, 1) = cloud->points[i].y;
       objectPoints.at<float>(i, 2) = cloud->points[i].z;
    }
    */

    cv::RNG rng(12345);
    std::vector<cv::Vec3b> colors;
    std::vector<int> labels(cloud->size());
    for (int j = 0; j < indices->cluster_indices.size(); j++) {
       std::vector<int> point_indices = indices->cluster_indices[j].indices;
       for (auto it = point_indices.begin(); it != point_indices.end(); it++) {
          int i = *it;
          objectPoints.at<float>(i, 0) = cloud->points[i].x;
          objectPoints.at<float>(i, 1) = cloud->points[i].y;
          objectPoints.at<float>(i, 2) = cloud->points[i].z;
          labels[i] = j;
       }
       colors.push_back(cv::Vec3b(rng.uniform(0, 255),
                                  rng.uniform(0, 255),
                                  rng.uniform(0, 255)));
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
    cv::Scalar color = cv::Scalar(0, 0, 0);
    cv::Mat image = cv::Mat(
       camera_info->height, camera_info->width, CV_8UC3, color);
    mask = cv::Mat::zeros(
       camera_info->height, camera_info->width, CV_32F);

    for (int i = 0; i < imagePoints.size(); i++) {
       int x = imagePoints[i].x;
       int y = imagePoints[i].y;
       if (!std::isnan(x) && !std::isnan(y) && (x >= 0 && x <= image.cols) &&
           (y >= 0 && y <= image.rows)) {
          
          /*
          image.at<cv::Vec3b>(y, x)[2] = cloud->points[i].r;
          image.at<cv::Vec3b>(y, x)[1] = cloud->points[i].g;
          image.at<cv::Vec3b>(y, x)[0] = cloud->points[i].b;
          */
          
          image.at<cv::Vec3b>(y, x) = colors[labels[i]];
          
          mask.at<float>(y, x) = 255.0f;
       }
    }
    return image;
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
    cv::Scalar color = cv::Scalar(0, 0, 0);
    cv::Mat image = cv::Mat(
       camera_info->height, camera_info->width, CV_8UC3, color);
    mask = cv::Mat::zeros(
       camera_info->height, camera_info->width, CV_32F);
    
    for (int i = 0; i < imagePoints.size(); i++) {
       int x = imagePoints[i].x;
       int y = imagePoints[i].y;
       if (!std::isnan(x) && !std::isnan(y) && (x >= 0 && x <= image.cols) &&
           (y >= 0 && y <= image.rows)) {
          image.at<cv::Vec3b>(y, x)[2] = cloud->points[i].r;
          image.at<cv::Vec3b>(y, x)[1] = cloud->points[i].g;
          image.at<cv::Vec3b>(y, x)[0] = cloud->points[i].b;

          mask.at<float>(y, x) = 255.0f;
       }
    }
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

cv::Rect PointCloudImageCreator::createMaskImages(
    const cv::Mat &image, cv::Mat &foreground, cv::Mat &background) {
    if (image.empty()) {
       return cv::Rect(0, 0, 0, 0);
    }
    foreground = cv::Mat::zeros(image.size(), CV_8UC1);
    background = cv::Mat(image.size(), CV_8UC1, cv::Scalar(255, 255, 255));
    for (int j = 0; j < image.rows; j++) {
       for (int i = 0; i < image.cols; i++) {
          if (image.at<cv::Vec3b>(j, i)[0] > 0) {
             foreground.at<uchar>(j, i) = 255;
             background.at<uchar>(j, i) = 0;
          }
       }
    }
    const int padding = 5;
    cv::Mat threshold_output = foreground.clone();
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(threshold_output, contours, hierarchy,
                     CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    int area = 0;
    cv::Rect rect = cv::Rect();
    for (int i = 0; i < contours.size(); i++) {
       cv::Rect a = cv::boundingRect(contours[i]);
        if (a.area() > area) {
            area = a.area();
            rect = a;
        }
    }
    rect.x = rect.x - padding;
    rect.y = rect.y - padding;
    rect.width = rect.width + (2 * padding);
    rect.height = rect.height + (2 * padding);

    if (this->is_roi_) {
      foreground = foreground(rect);
      background = background(rect);
    }
    return rect;
}

int main(int argc, char *argv[]) {

    ros::init(argc, argv, "point_cloud_image_creator");
    PointCloudImageCreator pcic;
    ros::spin();
}
