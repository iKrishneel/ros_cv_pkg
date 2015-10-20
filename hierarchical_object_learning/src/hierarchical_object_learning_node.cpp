// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo, Japan

#include <hierarchical_object_learning/hierarchical_object_learning.h>

HierarchicalObjectLearning::HierarchicalObjectLearning() {
    this->onInit();
}

void HierarchicalObjectLearning::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/hierarchical_object_learning/output/cloud", 1);
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/hierarchical_object_learning/output/image", 1);
    this->pub_pose_ = this->pnh_.advertise<geometry_msgs::PoseStamped>(
       "/hierarchical_object_learning/output/pose", 1);
}

void HierarchicalObjectLearning::subscribe() {
       this->sub_image_.subscribe(this->pnh_, "input_image", 1);
       this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
       this->sync_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy> >(100);
       sync_->connectInput(sub_image_, sub_cloud_);
       sync_->registerCallback(boost::bind(&HierarchicalObjectLearning::callback,
                                           this, _1, _2));
}

void HierarchicalObjectLearning::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_image_.unsubscribe();
}

void HierarchicalObjectLearning::callback(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    cv::Mat image = cv_bridge::toCvShare(
       image_msg, image_msg->encoding)->image;
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    cv_bridge::CvImage pub_img(
        image_msg->header, sensor_msgs::image_encodings::BGR8, image);
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;

    ROS_INFO("--Publish selected object info.");
    this->pub_cloud_.publish(ros_cloud);
    this->pub_image_.publish(pub_img.toImageMsg());
}

template<class T>
void HierarchicalObjectLearning::estimatePointCloudNormals(
    const pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<pcl::Normal>::Ptr normals,
    const T k, bool use_knn) const {
    if (cloud->empty()) {
      ROS_ERROR("ERROR: The Input cloud is Empty.....");
      return;
    }
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setNumberOfThreads(8);
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
    ne.setSearchMethod(tree);
    if (use_knn) {
      ne.setKSearch(k);
    } else {
      ne.setRadiusSearch(k);
    }
    ne.compute(*normals);
}

void HierarchicalObjectLearning::computePointFPFH(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    cv::Mat &histogram, bool holistic) const {
    if (cloud->empty() || normals->empty()) {
      ROS_ERROR("-- ERROR: cannot compute FPFH");
      return;
    }
    pcl::FPFHEstimationOMP<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    fpfh.setSearchMethod(tree);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(
        new pcl::PointCloud<pcl::FPFHSignature33> ());
    fpfh.setRadiusSearch(0.05);
    fpfh.compute(*fpfhs);
    const int hist_dim = 33;
    if (holistic) {
      histogram = cv::Mat::zeros(1, hist_dim, CV_32F);
      for (int i = 0; i < fpfhs->size(); i++) {
        for (int j = 0; j < hist_dim; j++) {
          histogram.at<float>(0, j) += fpfhs->points[i].histogram[j];
        }
      }
    } else {
      histogram = cv::Mat::zeros(
          static_cast<int>(fpfhs->size()), hist_dim, CV_32F);
      for (int i = 0; i < fpfhs->size(); i++) {
        for (int j = 0; j < hist_dim; j++) {
          histogram.at<float>(i, j) = fpfhs->points[i].histogram[j];
        }
      }
    }
    cv::normalize(histogram, histogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
}



int main(int argc, char *argv[]) {
    ros::init(argc, argv, "hierarchical_object_learning");
    HierarchicalObjectLearning hol;
    ros::spin();
    return 0;
}
