// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo, Japan

#include <hierarchical_object_learning/hierarchical_object_learning.h>

HierarchicalObjectLearning::HierarchicalObjectLearning() :
    num_threads_(8),
    cluster_size_(100),
    min_cloud_size_(100),
    neigbour_size_(16),
    downsize_(0.00f) {
    this->trainer_client_ = pnh_.serviceClient<
       hierarchical_object_learning::FitFeatureModel>("fit_feature_model");
   
    pnh_.getParam("source_type", this->source_type_);
    if (this->source_type_.compare("ROSBAG") == 0) {
       std::string rosbag_dir;
       pnh_.getParam("rosbag_directory", rosbag_dir);
       std::string topic;
       pnh_.getParam("sub_topic", topic);
       this->read_rosbag_file(rosbag_dir, topic);
    } else if (this->source_type_.compare("DETECTOR") == 0) {
       ROS_INFO("INITIALIZING ROS SUBSCRIBER");
       this->onInit();
    }
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
       this->sub_indices_.subscribe(this->pnh_, "input_indices", 1);
       this->sub_image_.subscribe(this->pnh_, "input_image", 1);
       this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
       this->sub_normals_.subscribe(this->pnh_, "input_normals", 1);
       this->sync_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy> >(100);
       sync_->connectInput(sub_image_, sub_cloud_, sub_normals_, sub_indices_);
       sync_->registerCallback(boost::bind(
                                  &HierarchicalObjectLearning::callback,
                                  this, _1, _2, _3, _4));
}

void HierarchicalObjectLearning::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_image_.unsubscribe();
}

void HierarchicalObjectLearning::callback(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::PointCloud2::ConstPtr &normal_msg,
    const jsk_recognition_msgs::ClusterPointIndices::ConstPtr &indices_msg) {
    cv::Mat image = cv_bridge::toCvShare(
       image_msg, image_msg->encoding)->image;
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::fromROSMsg(*normal_msg, *normals);
    
    std::vector<pcl::PointCloud<PointT>::Ptr> surfel_cloud;
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> surfel_normals;
    this->surfelsCloudFromIndices(cloud, normals, indices_msg,
                                  surfel_cloud, surfel_normals);

    sensor_msgs::CameraInfo::ConstPtr info_msg(new sensor_msgs::CameraInfo);
    if (surfel_cloud.size() == surfel_normals.size()) {
       for (int i = 0; i < surfel_cloud.size(); i++) {
          hierarchical_object_learning::FeatureArray surfel_featureMD;
          hierarchical_object_learning::FeatureArray point_featureMD;
          this->extractMultilevelCloudFeatures(*info_msg, cloud, normals,
                                       surfel_featureMD, point_featureMD,
                                       false);

          // classifier here
       }
    }
    
    cv_bridge::CvImage pub_img(
        image_msg->header, sensor_msgs::image_encodings::BGR8, image);
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;

    this->pub_cloud_.publish(ros_cloud);
    this->pub_image_.publish(pub_img.toImageMsg());
}

void HierarchicalObjectLearning::read_rosbag_file(
    const std::string path_to_bag,
    const std::string topic) {
    rosbag::Bag bag;
    bag.open(path_to_bag, rosbag::bagmode::Read);
    std::vector<std::string> topics;
    topics.push_back(topic);
    rosbag::View view(bag, rosbag::TopicQuery(topics));
    
    BOOST_FOREACH(rosbag::MessageInstance const m, view) {
       multilayer_object_tracking::ReferenceModelBundle::ConstPtr
          ref_bundle = m.instantiate<
             multilayer_object_tracking::ReferenceModelBundle>();
       
       jsk_recognition_msgs::PointsArray cloud_list(ref_bundle->cloud_bundle);
       sensor_msgs::Image image_msg(ref_bundle->image_bundle);
       sensor_msgs::CameraInfo info_msg(ref_bundle->cam_info);
       // cv::Mat image = cv_bridge::toCvShare(
       // image_msg, sensor_msgs::image_encodings::BGR8)->image;
       
       for (int i = 0; i < cloud_list.cloud_list.size(); i++) {
         sensor_msgs::PointCloud2 surfel_cloud(cloud_list.cloud_list[i]);
         pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
         pcl::fromROSMsg(surfel_cloud, *cloud);
         bool is_process_point = false;
         if (i == cloud_list.cloud_list.size() - 1 &&
             cloud->size() > this->min_cloud_size_) {
           is_process_point = true;
         }
         pcl::PointCloud<pcl::Normal>::Ptr normals(
            new pcl::PointCloud<pcl::Normal>);
         this->estimatePointCloudNormals<float>(
            cloud, normals, this->neigbour_size_, false);
         hierarchical_object_learning::FeatureArray surfel_featureMD;
         hierarchical_object_learning::FeatureArray point_featureMD;
         this->extractMultilevelCloudFeatures(info_msg, cloud, normals,
                                      surfel_featureMD, point_featureMD,
                                      is_process_point);

         // TODO(here): Make parallel
         {
           std::string save_surfel_model = "surfel_model";
           int is_success_surfel = this->fitFeatureModelService(
              surfel_featureMD, save_surfel_model);
           std::string save_point_model = "point_model";
           int is_success_point = this->fitFeatureModelService(
              point_featureMD, save_point_model);
         }
       }
    }
    bag.close();
    ROS_INFO("\033[35mSUCCESSFULLY COMPLETED\033[0m");
}

void HierarchicalObjectLearning::extractMultilevelCloudFeatures(
    const sensor_msgs::CameraInfo &info_msg,
    /*const sensor_msgs::Image &image_msg, */
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    hierarchical_object_learning::FeatureArray &surfel_feature_array,
    hierarchical_object_learning::FeatureArray &point_feature_array,
    bool is_point_level, bool is_surfel_level) {
    if (cloud->empty()) {
        ROS_ERROR("CANNOT PROCESS EMPTY CLOUD");
        return;
    }
    if (is_surfel_level) {
       jsk_recognition_msgs::Histogram surfel_features;
       this->extractObjectSurfelFeatures(cloud, normals, surfel_features);
       surfel_feature_array.feature_list.push_back(surfel_features);
    }
    
    if (is_point_level) {
      this->extractPointLevelFeatures(cloud, normals, point_feature_array,
                                      this->downsize_, this->cluster_size_);
    }
}

template<class T>
void HierarchicalObjectLearning::estimatePointCloudNormals(
    pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<pcl::Normal>::Ptr normals,
    const T k, bool use_knn) const {
    if (cloud->empty()) {
      ROS_ERROR("ERROR: The Input cloud is Empty.....");
      return;
    }
    std::vector<int> index;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, index);
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setNumberOfThreads(this->num_threads_);
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
    ne.setSearchMethod(tree);
    if (use_knn) {
      ne.setKSearch(k);
    } else {
      ne.setRadiusSearch(k);
    }
    ne.compute(*normals);
}

void HierarchicalObjectLearning::extractObjectSurfelFeatures(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    jsk_recognition_msgs::Histogram &histogram) {
    if (cloud->empty()) {
       ROS_ERROR("EMPTY CLOUD FOR MID-LEVEL FEATURES");
       return;
    }
    this->globalPointCloudFeatures(cloud, normals, histogram);
}

void HierarchicalObjectLearning::globalPointCloudFeatures(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    jsk_recognition_msgs::Histogram &histogram) {
    if (cloud->empty() || normals->empty()) {
      ROS_ERROR("ERROR: EMPTY CLOUD FOR SURFEL FEATURE");
      return;
    }
    pcl::VFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308> vfh;
    vfh.setInputCloud(cloud);
    vfh.setInputNormals(normals);
    pcl::search::KdTree<PointT>::Ptr tree(
       new pcl::search::KdTree<PointT>);
    vfh.setSearchMethod(tree);
    pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs(
       new pcl::PointCloud<pcl::VFHSignature308>());
    vfh.compute(*vfhs);
    const int feat_dim = 308;
    for (int i = 0; i < feat_dim; i++) {
       histogram.histogram.push_back(vfhs->points[0].histogram[i]);
    }
}

int HierarchicalObjectLearning::fitFeatureModelService(
    const hierarchical_object_learning::FeatureArray &feature_array,
    const std::string model_save_name) {
    hierarchical_object_learning::FitFeatureModel srv_ffm;
    srv_ffm.request.features = feature_array;
    srv_ffm.request.model_save_path = model_save_name;
    if (this->trainer_client_.call(srv_ffm)) {
       return srv_ffm.response.success;
    } else {
       ROS_ERROR("ERROR: FAILED TO CALL TRAINER SERIVCE");
       return false;
    }
}

void HierarchicalObjectLearning::extractPointLevelFeatures(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    hierarchical_object_learning::FeatureArray &feature_array,
    const float leaf_size, const int cluster_size) {
    if (cloud->empty()) {
       ROS_ERROR("EMPTY CLOUD FOR POINT-LEVEL FEATURES");
       return;
    }
    pcl::PointCloud<PointT>::Ptr keypoints_cloud(new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> grid;
    grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    grid.setInputCloud(cloud);
    grid.filter(*keypoints_cloud);
    
    this->computePointCloudFPFH(cloud, keypoints_cloud, normals, feature_array);

    
    
    // this->pointFeaturesBOWDescriptor(cloud, normals, featureMD, cluster_size);
    
    
}

void HierarchicalObjectLearning::pointFeaturesBOWDescriptor(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals, cv::Mat &vocabulary,
    const int cluster_size) {
    if (cloud->empty() || normals->empty()) {
       ROS_ERROR("CANNOT EXTRACT FEATURES OF EMPTY CLOUD");
       return;
    }
    cv::Mat geometric_features;
    cv::Mat color_features;
#ifdef _OPENMP
    #pragma omp parallel sections
#endif
    {
#ifdef _OPENMP
#pragma omp section
#endif
       {
          // this->computePointCloudFPFH(
          //    cloud, cloud, normals, geometric_features, true);
       }
#ifdef _OPENMP
#pragma omp section
#endif
       {
       this->pointIntensityFeature<int>(cloud, color_features, 8, true);
       }
    }
    cv::Mat feature_descriptor = cv::Mat(
       cloud->size(), color_features.cols + geometric_features.cols, CV_32F);
    cv::hconcat(geometric_features, color_features, feature_descriptor);

    // std::cout << "Feature Size: " << feature_descriptor.size() << std::endl;
    
    if (feature_descriptor.empty()) {
       return;
    }
    // cv::BOWKMeansTrainer bow_trainer(cluster_size);
    // bow_trainer.add(feature_descriptor);
    // vocabulary = bow_trainer.cluster();
    vocabulary = feature_descriptor.clone();
}

void HierarchicalObjectLearning::computePointCloudFPFH(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<PointT>::Ptr keypoints,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    hierarchical_object_learning::FeatureArray &feature_array,
    const float search_radius) const {
    if (cloud->empty() || normals->empty() || keypoints->empty()) {
      ROS_ERROR("-- ERROR: cannot compute FPFH");
      return;
    }
    pcl::FPFHEstimationOMP<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(keypoints);
    fpfh.setSearchSurface(cloud);
    fpfh.setInputNormals(normals);
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    fpfh.setSearchMethod(tree);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(
        new pcl::PointCloud<pcl::FPFHSignature33> ());
    fpfh.setRadiusSearch(search_radius);
    fpfh.compute(*fpfhs);
    const int hist_dim = 33;
    for (int i = 0; i < fpfhs->size(); i++) {
       jsk_recognition_msgs::Histogram histogram;
       for (int j = 0; j < hist_dim; j++) {
          histogram.histogram.push_back(fpfhs->points[i].histogram[j]);
       }
       feature_array.feature_list.push_back(histogram);
    }
}

template<class T>
void HierarchicalObjectLearning::pointIntensityFeature(
    const pcl::PointCloud<PointT>::Ptr cloud,
    cv::Mat &histogram, const T search_dim, bool is_knn) {
    if (cloud->empty()) {
       ROS_ERROR("-- ERROR: cannot compute FPFH");
       return;
    }
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);
    histogram = cv::Mat::zeros(static_cast<int>(cloud->size()), 3, CV_32F);

#ifdef _OPENMP
#pragma omp parallel for shared(histogram) num_threads(this->num_threads_)
#endif
    for (int j = 0; j < cloud->size(); j++) {
       std::vector<int> point_idx_search;
       std::vector<float> point_squared_distance;
       PointT pt = cloud->points[j];
       int search_out = 0;
       if (is_knn) {
          search_out = kdtree.nearestKSearch(
             pt, search_dim, point_idx_search, point_squared_distance);
       } else {
          search_out = kdtree.radiusSearch(
             pt, search_dim, point_idx_search, point_squared_distance);
       }
       float r_dist = 0.0;
       float g_dist = 0.0;
       float b_dist = 0.0;
       for (size_t i = 0; i < point_idx_search.size(); ++i) {
          r_dist += (pt.r - cloud->points[point_idx_search[i]].r)/255.0f;
          b_dist += (pt.b - cloud->points[point_idx_search[i]].b)/255.0f;
          g_dist += (pt.g - cloud->points[point_idx_search[i]].g)/255.0f;
       }
       histogram.at<float>(j, 0) = r_dist / static_cast<float>(
          point_idx_search.size());
       histogram.at<float>(j, 1) = g_dist / static_cast<float>(
          point_idx_search.size());
       histogram.at<float>(j, 2) = b_dist / static_cast<float>(
          point_idx_search.size());
    }
}

jsk_recognition_msgs::Histogram
HierarchicalObjectLearning::convertCvMatToFeatureMsg(
    const cv::Mat feature) {
    jsk_recognition_msgs::Histogram hist_msg;
    for (int j = 0; j < feature.rows; j++) {
       for (int i = 0; i < feature.cols; i++) {
          float val = feature.at<float>(j, i);
          hist_msg.histogram.push_back(val);
       }
    }
    return hist_msg;
}

void HierarchicalObjectLearning::surfelsCloudFromIndices(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    const jsk_recognition_msgs::ClusterPointIndices::ConstPtr &indices_msg,
    std::vector<pcl::PointCloud<PointT>::Ptr> &surfel_list,
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> &surfel_normals) {
    if (cloud->empty() || normals->empty()) {
      return;
    }
    pcl::ExtractIndices<PointT> extract_cloud;
    extract_cloud.setInputCloud(cloud);
    pcl::ExtractIndices<pcl::Normal> extract_normal;
    extract_normal.setInputCloud(normals);
    for (int i = 0; i < indices_msg->cluster_indices.size(); i++) {
      pcl::PointIndices::Ptr indices(new pcl::PointIndices);
      indices->indices = indices_msg->cluster_indices[i].indices;
      extract_cloud.setIndices(indices);
      extract_normal.setIndices(indices);
      extract_cloud.setNegative(false);
      extract_normal.setNegative(false);
      pcl::PointCloud<PointT>::Ptr surfel(new pcl::PointCloud<PointT>);
      extract_cloud.filter(*surfel);
      pcl::PointCloud<pcl::Normal>::Ptr normal(
         new pcl::PointCloud<pcl::Normal>);
      extract_normal.filter(*normal);
      if (surfel->size() > this->min_cloud_size_) {
         surfel_list.push_back(surfel);
         surfel_normals.push_back(normal);
      }
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "hierarchical_object_learning");
    HierarchicalObjectLearning hol;
    ros::spin();
    return 0;
}
