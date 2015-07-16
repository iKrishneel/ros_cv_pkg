// Copyright (C) 2015 by Krishneel Chaudhary, JSK Lab,
// The University of Tokyo, Japan

#include <multilayer_object_tracking/multilayer_object_tracking.h>

MultilayerObjectTracking::MultilayerObjectTracking() :
    init_counter_(0),
    min_cluster_size_(20),
    radius_search_(0.03f) {
    this->object_reference_ = ModelsPtr(new Models);
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
    this->sub_obj_cloud_.subscribe(this->pnh_, "input_obj_cloud", 1);
    this->sub_obj_indices_.subscribe(this->pnh_, "input_obj_indices", 1);
    this->obj_sync_ = boost::make_shared<message_filters::Synchronizer<
       ObjectSyncPolicy> >(100);
    obj_sync_->connectInput(sub_obj_indices_, sub_obj_cloud_);
    obj_sync_->registerCallback(boost::bind(
                               &MultilayerObjectTracking::objInitCallback,
                               this, _1, _2));
    
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

void MultilayerObjectTracking::objInitCallback(
    const jsk_recognition_msgs::ClusterPointIndicesConstPtr &indices_mgs,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    std::vector<pcl::PointIndices::Ptr> cluster_indices;
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    cluster_indices = this->clusterPointIndicesToPointIndices(indices_mgs);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (this->init_counter_++ > 0) {
       // reset tracking params
       ROS_WARN("Object is re-initalized! stopping & reseting...");
    }
    if (!cloud->empty()) {
       std::vector<pcl::PointIndices::Ptr> all_indices =
          this->clusterPointIndicesToPointIndices(indices_mgs);
       this->object_reference_ = ModelsPtr(new Models);
       for (std::vector<pcl::PointIndices::Ptr>::iterator it =
               all_indices.begin(); it != all_indices.end(); it++) {
          if ((*it)->indices.size() > this->min_cluster_size_) {
             ReferenceModel ref_model;
             ref_model.cloud_clusters = pcl::PointCloud<PointT>::Ptr(
                new pcl::PointCloud<PointT>);
             ref_model.cluster_normals = pcl::PointCloud<pcl::Normal>::Ptr(
                new pcl::PointCloud<pcl::Normal>);
             pcl::ExtractIndices<PointT>::Ptr eifilter(
                new pcl::ExtractIndices<PointT>);
             eifilter->setInputCloud(cloud);
             eifilter->setIndices(*it);
             eifilter->filter(*ref_model.cloud_clusters);
             this->estimatePointCloudNormals<float>(
                ref_model.cloud_clusters,
                ref_model.cluster_normals,
                this->radius_search_);
             this->computeCloudClusterRPYHistogram(
                ref_model.cloud_clusters,
                ref_model.cluster_normals,
                ref_model.cluster_vfh_hist);
             this->computeColorHistogram(
                ref_model.cloud_clusters,
                ref_model.cluster_color_hist);

             std::cout << "DEBUG: Model Info: "
                       << ref_model.cloud_clusters->size() << "\t"
                       << ref_model.cluster_normals->size() << "\t"
                       << ref_model.cluster_vfh_hist.size() << "\t"
                       << ref_model.cluster_color_hist.size()
                       << std::endl;
          
             this->object_reference_->push_back(ref_model);
          }
       }
    }
}

void MultilayerObjectTracking::callback(
    const jsk_recognition_msgs::ClusterPointIndicesConstPtr &indices_mgs,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    if (this->object_reference_->empty()) {
       ROS_WARN("No Model To Track Selected");
       return;
    }
   
    std::vector<pcl::PointIndices::Ptr> all_indices;
    all_indices = this->clusterPointIndicesToPointIndices(indices_mgs);

    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    this->globalLayerPointCloudProcessing(cloud, all_indices);

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

void MultilayerObjectTracking::globalLayerPointCloudProcessing(
    pcl::PointCloud<PointT>::Ptr cloud,
    const std::vector<pcl::PointIndices::Ptr> &all_indices) {
    if (cloud->empty() || all_indices.empty()) {
       ROS_ERROR("ERROR: Global Layer Input Empty");
       return;
    }
    pcl::PointCloud<PointT>::Ptr n_cloud(new pcl::PointCloud<PointT>);
    Models obj_ref = *object_reference_;
    for (std::vector<pcl::PointIndices::Ptr>::const_iterator it =
            all_indices.begin(); it != all_indices.end(); it++) {
       if ((*it)->indices.size() > this->min_cluster_size_) {
          pcl::ExtractIndices<PointT>::Ptr eifilter(
             new pcl::ExtractIndices<PointT>);
          eifilter->setInputCloud(cloud);
          eifilter->setIndices(*it);
          pcl::PointCloud<PointT>::Ptr cloud_cluster(
             new pcl::PointCloud<PointT>);
          eifilter->filter(*cloud_cluster);
          pcl::PointCloud<pcl::Normal>::Ptr cluster_normal(
             new pcl::PointCloud<pcl::Normal>);
          this->estimatePointCloudNormals<float>(
             cloud_cluster, cluster_normal, this->radius_search_);
          cv::Mat cluster_hist;
          this->computeCloudClusterRPYHistogram(
             cloud_cluster, cluster_normal, cluster_hist);
          cv::Mat color_hist;
          this->computeColorHistogram(cloud_cluster, color_hist);
          float probability = 0.0;
          for (int i = 0; i < obj_ref.size(); i++) {
             float dist_vfh = static_cast<float>(
                cv::compareHist(cluster_hist,
                                obj_ref[i].cluster_vfh_hist,
                                CV_COMP_BHATTACHARYYA));
             float dist_col = static_cast<float>(
                cv::compareHist(color_hist,
                                obj_ref[i].cluster_color_hist,
                                CV_COMP_BHATTACHARYYA));
             
             float prob = std::exp(-0.7 * dist_vfh) * std::exp(-1.5 * dist_col);
             if (prob > probability /*&& prob > 0.5f*/) {
                probability = prob;
             }
          }
          for (int i = 0; i < cloud_cluster->size(); i++) {
             PointT pt = cloud_cluster->points[i];
             pt.r = 255 * probability;
             pt.g = 255 * probability;
             pt.b = 255 * probability;
             n_cloud->push_back(pt);
          }
       }
    }
    cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*n_cloud, *cloud);
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

void MultilayerObjectTracking::computeCloudClusterRPYHistogram(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normal,
    cv::Mat &histogram) {
    if (cloud->empty() || normal->empty()) {
       ROS_ERROR("ERROR: Empty Input");
       return;
    }
    pcl::VFHEstimation<PointT,
                       pcl::Normal,
                       pcl::VFHSignature308> vfh;
    vfh.setInputCloud(cloud);
    vfh.setInputNormals(normal);
    pcl::search::KdTree<PointT>::Ptr tree(
       new pcl::search::KdTree<PointT>);
    vfh.setSearchMethod(tree);
    pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs(
       new pcl::PointCloud<pcl::VFHSignature308>());
    vfh.compute(*vfhs);
    histogram = cv::Mat(sizeof(char), 308, CV_32F);
    for (int i = 0; i < histogram.cols; i++) {
       histogram.at<float>(0, i) = vfhs->points[0].histogram[i];
    }
    cv::normalize(histogram, histogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
}

void MultilayerObjectTracking::computeColorHistogram(
    const pcl::PointCloud<PointT>::Ptr cloud,
    cv::Mat &hist, const int hBin, const int sBin, bool is_norm) {
    cv::Mat pixels = cv::Mat::zeros(
       sizeof(char), static_cast<int>(cloud->size()), CV_8UC3);
    for (int i = 0; i < cloud->size(); i++) {
       cv::Vec3b pix_val;
       pix_val[0] = cloud->points[i].b;
       pix_val[1] = cloud->points[i].g;
       pix_val[2] = cloud->points[i].r;
       pixels.at<cv::Vec3b>(0, i) = pix_val;
    }
    cv::Mat hsv;
    cv::cvtColor(pixels, hsv, CV_BGR2HSV);
    int histSize[] = {hBin, sBin};
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    const float* ranges[] = {h_ranges, s_ranges};
    int channels[] = {0, 1};
    cv::calcHist(
       &hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    if (is_norm) {
       cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    }
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
    for (int i = 0; i < indices_mgs->cluster_indices.size(); i++) {
       std::vector<int> indices = indices_mgs->cluster_indices[i].indices;
       pcl::PointIndices::Ptr pcl_indices (new pcl::PointIndices);
       pcl_indices->indices = indices;
       ret.push_back(pcl_indices);
    }
    return ret;
}

int main(int argc, char *argv[]) {

    ros::init(argc, argv, "multilayer_object_tracking");
    MultilayerObjectTracking cpi2i;
    ros::spin();
    return 0;
}
