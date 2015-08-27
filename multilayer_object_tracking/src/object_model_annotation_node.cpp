// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo, Japan

#include <multilayer_object_tracking/object_model_annotation.h>
#include <algorithm>

ObjectModelAnnotation::ObjectModelAnnotation() {
    this->onInit();
}

void ObjectModelAnnotation::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/object_model/output/cloud", 1);
    this->pub_background_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/object_model/output/bkgd_cloud", 1);
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/object_model/output/image", 1);
    this->pub_pose_ = this->pnh_.advertise<geometry_msgs::PoseStamped>(
       "/object_model/output/pose", 1);
}

void ObjectModelAnnotation::subscribe() {
       this->sub_image_.subscribe(this->pnh_, "input_image", 1);
       this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
       this->sub_screen_pt_.subscribe(this->pnh_, "input_screen", 1);
       this->sync_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy> >(100);
       sync_->connectInput(sub_image_, sub_cloud_, sub_screen_pt_);
       sync_->registerCallback(boost::bind(&ObjectModelAnnotation::callback,
                                           this, _1, _2, _3));
}

void ObjectModelAnnotation::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_image_.unsubscribe();
}


void ObjectModelAnnotation::callback(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const geometry_msgs::PolygonStampedConstPtr &screen_msg) {
    // boost::mutex::scoped_lock lock(this->mutex_);
    int x = screen_msg->polygon.points[0].x;
    int y = screen_msg->polygon.points[0].y;
    int width = screen_msg->polygon.points[1].x - x;
    int height = screen_msg->polygon.points[1].y - y;
    cv::Rect_<int> screen_rect = cv::Rect_<int>(x, y, width, height);
    if (width < 24 && height < 24) {
       ROS_WARN("-- Selected Object Size is too small... Not init tracker");
       return;
    }
    cv::Mat image = cv_bridge::toCvShare(
       image_msg, image_msg->encoding)->image;
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    pcl::PointCloud<PointT>::Ptr bg_cloud(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(*cloud, *bg_cloud);
    this->getAnnotatedObjectCloud(cloud, image, screen_rect);
    Eigen::Vector4f centroid;
    this->compute3DCentroids(cloud, centroid);
    this->backgroundPointCloudIndices(
       bg_cloud, cloud, centroid, image.size(), screen_rect);
    geometry_msgs::PoseStamped ros_pose;
    ros_pose.pose.position.x = centroid(0);
    ros_pose.pose.position.y = centroid(1);
    ros_pose.pose.position.z = centroid(2);
    ros_pose.pose.orientation.x = 0.0f;
    ros_pose.pose.orientation.y = 0.0f;
    ros_pose.pose.orientation.z = 0.0f;
    ros_pose.pose.orientation.w = 0.0f;
    ros_pose.header = cloud_msg->header;
    
    cv_bridge::CvImage pub_img(
       image_msg->header, sensor_msgs::image_encodings::BGR8, image);
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;

    sensor_msgs::PointCloud2 ros_bkgd;
    pcl::toROSMsg(*bg_cloud, ros_bkgd);
    ros_bkgd.header = cloud_msg->header;

    ROS_INFO("--Publish selected object info.");
    this->pub_cloud_.publish(ros_cloud);
    this->pub_image_.publish(pub_img.toImageMsg());
    this->pub_pose_.publish(ros_pose);
    this->pub_background_.publish(ros_bkgd);
}

void ObjectModelAnnotation::getAnnotatedObjectCloud(
    pcl::PointCloud<PointT>::Ptr cloud,
    const cv::Mat &image,
    const cv::Rect_<int> screen_rect) {
    if (cloud->empty() || image.empty()) {
       ROS_ERROR("-- Cannot Process Empty Cloud");
       return;
    }
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    this->imageToPointCloudIndices(cloud, indices, image.size(), screen_rect);
    pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(*cloud, *tmp_cloud);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> euclidean_clustering;
    euclidean_clustering.setClusterTolerance(0.01);
    euclidean_clustering.setMinClusterSize(60);
    euclidean_clustering.setMaxClusterSize(25000);
    euclidean_clustering.setSearchMethod(tree);
    euclidean_clustering.setInputCloud(cloud);
    euclidean_clustering.setIndices(indices);
    euclidean_clustering.extract(cluster_indices);
    int max_size = FLT_MIN;
    int index = 0;
    for (int i = 0; i < cluster_indices.size(); i++) {
       int c_size = cluster_indices[i].indices.size();
       if (c_size > max_size) {
          max_size = c_size;
          index = i;
       }
    }
    cloud->clear();
    for (int i = 0; i < cluster_indices[index].indices.size(); i++) {
       int pi = cluster_indices[index].indices[i];
       cloud->points.push_back(tmp_cloud->points[pi]);
    }
}

void ObjectModelAnnotation::imageToPointCloudIndices(
    pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointIndices::Ptr indices,
    const cv::Size size, const cv::Rect_<int> rect) {
    if (cloud->empty()) {
       ROS_ERROR("-- Cannot Process Empty Cloud");
       return;
    }
    for (int j = rect.y; j < (rect.y + rect.height); j++) {
       for (int i = rect.x; i < (rect.x + rect.width); i++) {
          int index = i + (j * size.width);
          PointT pt = cloud->points[index];
          if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z)) {
             indices->indices.push_back(index);
          }
       }
    }
    for (int i = 0; i < cloud->size(); i++) {
       PointT pt = cloud->points[i];
       if (isnan(pt.x) || isnan(pt.y) || isnan(pt.z)) {
          pt.x = 0;
          pt.y = 0;
          pt.z = 0;
          cloud->points[i] = pt;
       }
    }
}

void ObjectModelAnnotation::backgroundPointCloudIndices(
    pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<PointT>::Ptr template_cloud,
    const Eigen::Vector4f centroid,
    const cv::Size size, const cv::Rect_<int> rect) {
    if (cloud->empty()) {
        ROS_ERROR("-- Cannot Process Empty Cloud");
        return;
    }
    float lenght = std::max(static_cast<float>(rect.width),
                            static_cast<float>(rect.height));
    lenght /= 1.0f;
    int center_x = rect.x + rect.width/2;
    int center_y = rect.y + rect.height/2;
    int min_x = center_x - lenght;
    min_x = ((min_x < 0) ? 0 : min_x);
    int min_y = center_y - lenght;
    min_y = ((min_y < 0) ? 0 : min_y);
    int n_width = 2 * lenght;
    n_width = ((n_width + min_x > size.width) ?
               size.width - min_x : n_width);
    int n_height = 2 * lenght;
    n_height = ((n_height + min_y > size.height) ?
                size.height - min_y : n_height);
    cv::Rect_<int> bkgd_rect = cv::Rect_<int>(
       min_x, min_y, n_width, n_height);
    pcl::PointCloud<PointT>::Ptr bkgd_cloud(new pcl::PointCloud<PointT>);
    for (int j = bkgd_rect.y; j < (bkgd_rect.y + bkgd_rect.height); j++) {
       for (int i = bkgd_rect.x; i < (bkgd_rect.x + bkgd_rect.width); i++) {
          if ((j > rect.y && j < rect.y + rect.height) &&
              (i > rect.x && i < rect.x + rect.width)) {
             continue;
          } else {
             int index = i + (j * size.width);
             PointT pt = cloud->points[index];
             if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z)) {
                bkgd_cloud->push_back(pt);
             }
          }
       }
    }
    bool is_filter = this->filterPointCloud(bkgd_cloud,
                                            template_cloud,
                                            centroid);
    if (!is_filter) {
       ROS_ERROR("FILTER ERROR");
       return;
    }
    cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*bkgd_cloud, *cloud);
}

float ObjectModelAnnotation::templateCloudFilterLenght(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const Eigen::Vector4f centroid) {
    if (cloud->empty()) {
        ROS_ERROR("ERROR! Input Cloud is Empty");
        return -1.0f;
    }
    Eigen::Vector4f pivot_pt = centroid;
    Eigen::Vector4f max_pt;
    pcl::getMaxDistance<PointT>(*cloud, pivot_pt, max_pt);
    pivot_pt(3) = 0.0f;
    max_pt(3) = 0.0f;
    float dist = static_cast<float>(pcl::distances::l2(max_pt, pivot_pt));
    return (dist);
}

bool ObjectModelAnnotation::filterPointCloud(
    pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<PointT>::Ptr template_cloud,
    const Eigen::Vector4f centroid,
    const float scaling_factor) {
    if (cloud->empty() || template_cloud->empty()) {
        ROS_ERROR("ERROR! Input data is empty is Empty");
        return false;
    }
    float filter_distance = this->templateCloudFilterLenght(
       template_cloud, centroid);
    filter_distance *= scaling_factor;
    if (filter_distance < 0.1f) {
        return false;
    }
    pcl::PointCloud<PointT>::Ptr cloud_filter(new pcl::PointCloud<PointT>);
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    float min_x = centroid(0) - filter_distance;
    float max_x = centroid(0) + filter_distance;
    pass.setFilterLimits(min_x, max_x);
    pass.filter(*cloud_filter);
    pass.setInputCloud(cloud_filter);
    pass.setFilterFieldName("y");
    float min_y = centroid(1) - filter_distance;
    float max_y = centroid(1) + filter_distance;
    pass.setFilterLimits(min_y, max_y);
    pass.filter(*cloud_filter);
    pass.setInputCloud(cloud_filter);
    pass.setFilterFieldName("z");
    float min_z = centroid(2) - filter_distance;
    float max_z = centroid(2) + filter_distance;
    pass.setFilterLimits(min_z, max_z);
    pass.filter(*cloud_filter);
    if (cloud_filter->empty()) {
        return false;
    }
    cloud->empty();
    pcl::copyPointCloud<PointT, PointT>(*cloud_filter, *cloud);
    return true;
}


void ObjectModelAnnotation::compute3DCentroids(
    const pcl::PointCloud<PointT>::Ptr cloud,
    Eigen::Vector4f &centre) {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: empty cloud for centroid");
       centre = Eigen::Vector4f(-1, -1, -1, -1);
       return;
    }
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid<PointT, float>(*cloud, centroid);
    if (!isnan(centroid(0)) && !isnan(centroid(1)) && !isnan(centroid(2))) {
       centre = centroid;
    } else {
       ROS_ERROR("ERROR: NAN CENTROID\n-- reselect object");
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "object_model_annotation");
    ObjectModelAnnotation oma;
    ros::spin();
    return 0;
}
