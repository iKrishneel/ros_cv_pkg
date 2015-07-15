// Copyright (C) 2015 by Krishneel Chaudhary, JSK Lab,
// The University of Tokyo, Japan

#include <multilayer_object_tracking/multilayer_object_tracking.h>

MultilayerObjectTracking::MultilayerObjectTracking() {   
    this->model_cloud_ = pcl::PointCloud<PointT>::Ptr(
       new pcl::PointCloud<PointT>);
    this->loadModelPcdFile(this->model_cloud_, "flake_box.pcd");    
    this->subscribe();
    this->onInit();
}

void MultilayerObjectTracking::onInit() {
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/cluster_point_indices_to_image/output/cloud", 1);
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/cluster_point_indices_to_image/output/image", 1);
}

void MultilayerObjectTracking::subscribe() {
    this->sub_select_.subscribe(this->pnh_, "selected_region", 1);
    this->sub_indices_.subscribe(this->pnh_, "input_indices", 1);
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    sync_->connectInput(sub_indices_, sub_cloud_);
    sync_->registerCallback(boost::bind(
                               &MultilayerObjectTracking::callback,
                               this, _1, _2));
}

void MultilayerObjectTracking::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_indices_.unsubscribe();
}

void MultilayerObjectTracking::callback(
    const jsk_recognition_msgs::ClusterPointIndicesConstPtr &indices_mgs,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {

    // std::vector<pcl::PointIndices::Ptr> cluster_indices;
    // cluster_indices = this->clusterPointIndicesToPointIndices(indices_mgs);


    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    std::vector<int> index;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, index);
    
    pcl::PointCloud<pcl::Normal>::Ptr normals(
       new pcl::PointCloud<pcl::Normal>);
    this->estimatePointCloudNormals(cloud, normals, 0.03, false);
    cv::Mat scene_fpfh;
    this->computePointFPFH(cloud, normals, scene_fpfh, false);

    std::cout << "-- computing weight" << std::endl;
    
    for (int i = 0; i < scene_fpfh.rows; i++) {
       float dist = static_cast<float>(
          cv::compareHist(
             this->model_fpfh_, scene_fpfh.row(i), CV_COMP_BHATTACHARYYA));
       // float weight = exp(-1 * dist);
       if (dist > 0.5f) {
          cloud->points[i].r = 0;
          cloud->points[i].g = 0;
          cloud->points[i].b = 0;
       }
    }
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

template<class T>
void MultilayerObjectTracking::estimatePointCloudNormals(
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

void MultilayerObjectTracking::computePointFPFH(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    cv::Mat &histogram, bool holistic) {
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

std::vector<pcl::PointIndices::Ptr>
MultilayerObjectTracking::clusterPointIndicesToPointIndices(
    const jsk_recognition_msgs::ClusterPointIndicesConstPtr &indices_mgs) {
    std::vector<pcl::PointIndices::Ptr> ret;
    int icounter = 0;
    for (int i = 0; i < indices_mgs->cluster_indices.size(); i++) {
       std::vector<int> indices = indices_mgs->cluster_indices[i].indices;
       pcl::PointIndices::Ptr pcl_indices (new pcl::PointIndices);
       pcl_indices->indices = indices;
       ret.push_back(pcl_indices);
       icounter += indices.size();
    }
    std::cout << "Size: " << icounter  << std::endl;
    
    return ret;
}

void MultilayerObjectTracking::loadModelPcdFile(
    pcl::PointCloud<PointT>::Ptr cloud,
    const char* filename) {
    int load = pcl::io::loadPCDFile<PointT>(filename, *cloud);
    if (load == -1) {
       ROS_ERROR("-- %s not such file found!", filename);
       return;
    } else {
       std::cout << "Successfully loaded PCD" << std::endl;
       std::cout << "Size: " << cloud->size() << std::endl;

       pcl::PointCloud<pcl::Normal>::Ptr normals(
          new pcl::PointCloud<pcl::Normal>);
       this->estimatePointCloudNormals<float>(cloud, normals, 0.03f, false);
       this->computePointFPFH(cloud, normals, this->model_fpfh_);
    }
}

int main(int argc, char *argv[]) {

    ros::init(argc, argv, "multilayer_object_tracking");
    MultilayerObjectTracking cpi2i;
    ros::spin();
    return 0;
}
