
#include <point_cloud_cluster_matching/point_cloud_cluster_matching.h>

PointCloudClusterMatching::PointCloudClusterMatching() {
    this->subscribe();
    this->onInit();
}

void PointCloudClusterMatching::onInit() {

    this->pub_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>(
       "/model/output/cloud_cluster", sizeof(char));
    
    this->pub_signal_ = pnh_.advertise<std_msgs::Bool>(
       "/point_cloud_cluster_matching/output/signal", sizeof(char));
}

void PointCloudClusterMatching::subscribe() {

    // this->sub_signal_ = this->pnh_.subscribe(
    //    "input_signal", sizeof(char),
    //    &PointCloudClusterMatching::signalCallback, this);
    this->sub_indices_ = this->pnh_.subscribe(
       "input_indices", sizeof(char),
       &PointCloudClusterMatching::indicesCallback, this);
    this->sub_cloud_prev_ = this->pnh_.subscribe(
      "input_cloud_prev", sizeof(char),
      &PointCloudClusterMatching::cloudPrevCallback, this);
    this->sub_cam_info_ = this->pnh_.subscribe(
      "input_info", sizeof(char),
      &PointCloudClusterMatching::cameraInfoCallback, this);
    this->sub_cloud_ = this->pnh_.subscribe(
      "input_cloud", sizeof(char),
      &PointCloudClusterMatching::cloudCallback, this);
   
}

void PointCloudClusterMatching::unsubscribe() {
    this->sub_cloud_.shutdown();
    this->sub_indices_.shutdown();
    this->sub_signal_.shutdown();
    this->sub_manip_cluster_.shutdown();
    this->sub_grip_end_pose_.shutdown();
}

void PointCloudClusterMatching::signalCallback(
    const std_msgs::Bool &signal_msg) {
    if (signal_msg.data) {
       this->sub_grip_end_pose_ = this->pnh_.subscribe(
          "input_gripper_end_pose", sizeof(char),
          &PointCloudClusterMatching::gripperEndPoseCallback, this);
       this->sub_manip_cluster_ = this->pnh_.subscribe(
          "input_manip_cluster", sizeof(char),
          &PointCloudClusterMatching::manipulatedClusterCallback, this);
       this->sub_indices_ = this->pnh_.subscribe(
          "input_indices", sizeof(char),
          &PointCloudClusterMatching::indicesCallback, this);
       this->sub_cloud_prev_ = this->pnh_.subscribe(
          "input_cloud_prev", sizeof(char),
          &PointCloudClusterMatching::cloudPrevCallback, this);
       this->sub_cloud_ = this->pnh_.subscribe(
          "input_cloud", sizeof(char),
          &PointCloudClusterMatching::cloudCallback, this);
       this->sub_cam_info_ = this->pnh_.subscribe(
          "input_info", sizeof(char),
          &PointCloudClusterMatching::cameraInfoCallback, this);
    } else {
       std_msgs::Bool next_step;
       next_step.data = false;
       this->pub_signal_.publish(next_step);
    }
}

void PointCloudClusterMatching::gripperEndPoseCallback(
    const geometry_msgs::Pose & end_pose_msg) {
    this->gripper_pose_ = end_pose_msg;
}

void PointCloudClusterMatching::manipulatedClusterCallback(
    const std_msgs::Int16 &manip_cluster_index_msg) {
    this->manipulated_cluster_index_ = manip_cluster_index_msg.data;
}

void PointCloudClusterMatching::cameraInfoCallback(
    const sensor_msgs::CameraInfo::ConstPtr &info_msg) {
    boost::mutex::scoped_lock lock(this->lock_);
    camera_info_ = info_msg;
}

void PointCloudClusterMatching::indicesCallback(
    const jsk_recognition_msgs::ClusterPointIndices &indices_msgs) {
    this->all_indices.clear();
    for (int i = 0; i < indices_msgs.cluster_indices.size(); i++) {
       pcl::PointIndices indices;
       indices.indices = indices_msgs.cluster_indices[i].indices;
       this->all_indices.push_back(indices);
    }
}

void PointCloudClusterMatching::cloudPrevCallback(
    const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud_prev(
      new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud_prev);
    this->prev_cloud_clusters.clear();
    this->objectCloudClusters(
       cloud_prev, this->all_indices, this->prev_cloud_clusters);
}


void PointCloudClusterMatching::cloudCallback(
    const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty() || this->prev_cloud_clusters.empty()) {
       ROS_ERROR("-- EMPTY CLOUD CANNOT BE PROCESSED.");
       return;
    }
    
}

void PointCloudClusterMatching::createImageFromObjectClusters(
    const std::vector<pcl::PointCloud<PointT>::Ptr> &cloud_clusters,
    const sensor_msgs::CameraInfo::ConstPtr camera_info,
    std::vector<cv::Mat> &image_patches) {
    image_patches.clear();
    for (std::vector<pcl::PointCloud<PointT>::Ptr>::const_iterator it =
            cloud_clusters.begin(); it != cloud_clusters.end(); it++) {
       if (!(*it)->empty()) {
          cv::Mat mask;
          cv::Mat img_out = this->projectPointCloudToImagePlane(
             *it, camera_info, mask);
          cv::Mat interpolate_img = this->interpolateImage(img_out, mask);
          
       }
    }
}



void PointCloudClusterMatching::objectCloudClusters(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const std::vector<pcl::PointIndices> &cluster_indices,
    std::vector<pcl::PointCloud<PointT>::Ptr> &cloud_clusters) {
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    for (std::vector<pcl::PointIndices>::const_iterator it =
            cluster_indices.begin(); it != cluster_indices.end(); it++) {
       pcl::PointCloud<PointT>::Ptr c_cloud(new pcl::PointCloud<PointT>);
       pcl::PointIndices::Ptr indices(new pcl::PointIndices());
       indices->indices = it->indices;
       extract.setIndices(indices);
       extract.setNegative(false);
       extract.filter(*c_cloud);
       cloud_clusters.push_back(c_cloud);
    }
}




















void PointCloudClusterMatching::extractFeaturesAndMatchCloudPoints(
    const pcl::PointCloud<PointT>::Ptr model_cloud,
    const pcl::PointCloud<PointT>::Ptr scene_cloud,
    AffineTrans &rototranslations,
    std::vector<pcl::Correspondences> &clustered_corrs) {
    if (model_cloud->empty() || scene_cloud->empty()) {
       ROS_ERROR("-- EMPTY CLOUD CANNOT BE MATCHED");
       return;
    }
    // extract cloud normals
    pcl::PointCloud<pcl::Normal>::Ptr model_normals(
       new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals(
       new pcl::PointCloud<pcl::Normal>);
    this->pointCloudNormal(model_cloud, model_normals);
    this->pointCloudNormal(scene_cloud, scene_normals);

    
    // extract cloud keypoints
    const float search_radius = 0.01f;
    pcl::PointCloud<PointT>::Ptr model_keypoints(
       new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr scene_keypoints(
       new pcl::PointCloud<PointT>);
    this->getCloudClusterKeyPoints(model_cloud, model_keypoints, search_radius);
    this->getCloudClusterKeyPoints(scene_cloud, scene_keypoints, search_radius);

    
    // extract keypoint descriptors
    const float descr_search_radius = 0.02f;
    pcl::PointCloud<DescriptorType>::Ptr model_descriptors;
    pcl::PointCloud<DescriptorType>::Ptr scene_descriptors;
    this->computeDescriptors(model_cloud, model_keypoints, model_normals,
                             model_descriptors, descr_search_radius);
    this->computeDescriptors(scene_cloud, scene_keypoints, scene_normals,
                             scene_descriptors, descr_search_radius);

    /*
    // model - scene correspondance
    const float matching_threshold = 0.25f;
    pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());
    this->modelSceneCorrespondences(model_descriptors, scene_descriptors,
                                    model_scene_corrs, matching_threshold);

    // clustering
    const float rf_radius = 0.015f;
    const float hough_bin_size = 0.01f;
    const float hough_threshold = 5.0f;
    this->HoughCorrespondanceClustering(
       model_cloud, model_normals, model_keypoints,
       scene_cloud, scene_normals, scene_keypoints,
       model_scene_corrs, rototranslations, clustered_corrs,
       rf_radius, hough_bin_size, hough_threshold);
    */
}

void PointCloudClusterMatching::pointCloudNormal(
    const pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<pcl::Normal>::Ptr normals) {
    if (cloud->empty()) {
       ROS_ERROR("-- EMPTY CLOUD");
       return;
    }
    pcl::NormalEstimationOMP<PointT, pcl::Normal> norm_est;
    norm_est.setKSearch(10);
    norm_est.setInputCloud(cloud);
    norm_est.compute(*normals);
}

void PointCloudClusterMatching::getCloudClusterKeyPoints(
    const pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<PointT>::Ptr cloud_keypoints,
    const float search_radius) {
    pcl::PointCloud<int> sampled_indices;
    pcl::UniformSampling<PointT> uniform_sampling;
    uniform_sampling.setInputCloud(cloud);
    uniform_sampling.setRadiusSearch(search_radius);
    uniform_sampling.compute(sampled_indices);
    pcl::copyPointCloud(*cloud, sampled_indices.points, *cloud_keypoints);
}

void PointCloudClusterMatching::computeDescriptors(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<PointT>::Ptr cloud_keypoints,
    const pcl::PointCloud<pcl::Normal>::Ptr cloud_normals,
    pcl::PointCloud<DescriptorType>::Ptr cloud_descriptor,
    const float search_radius) {
    pcl::SHOTEstimationOMP<PointT, pcl::Normal, DescriptorType>::Ptr
       descriptr_est (new pcl::SHOTEstimationOMP<PointT, pcl::Normal, DescriptorType>);
    descriptr_est->setRadiusSearch(search_radius);
    descriptr_est->setInputNormals(cloud_normals);
    descriptr_est->setSearchSurface(cloud);
    descriptr_est->compute(*cloud_descriptor);
}

void PointCloudClusterMatching::modelSceneCorrespondences(
    pcl::PointCloud<DescriptorType>::Ptr model_descriptors,
    pcl::PointCloud<DescriptorType>::Ptr scene_descriptors,
    pcl::CorrespondencesPtr model_scene_corrs,
    const float threshold) {
   // pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());
    pcl::KdTreeFLANN<DescriptorType> match_search;
    match_search.setInputCloud(model_descriptors);
    for (int i = 0; i < scene_descriptors->size(); i++) {
       std::vector<int> neigh_indices(1);
       std::vector<float> neigh_sqr_dists(1);
       if (!isnan(scene_descriptors->at(i).descriptor[0])) {
          int found_neighs = match_search.nearestKSearch(
             scene_descriptors->at(i), 1, neigh_indices, neigh_sqr_dists);
          if (found_neighs == 1 && neigh_sqr_dists[0] < threshold) {
             pcl::Correspondence corr(neigh_indices[0], i, neigh_sqr_dists[0]);
             model_scene_corrs->push_back(corr);
          }
       }
    }
}

void PointCloudClusterMatching::HoughCorrespondanceClustering(
    const pcl::PointCloud<PointT>::Ptr model,
    const pcl::PointCloud<pcl::Normal>::Ptr model_normals,
    const pcl::PointCloud<PointT>::Ptr model_keypoints,
    const pcl::PointCloud<PointT>::Ptr scene,
    const pcl::PointCloud<pcl::Normal>::Ptr scene_normals,
    const pcl::PointCloud<PointT>::Ptr scene_keypoints,
    pcl::CorrespondencesPtr model_scene_corrs,
    AffineTrans &rototranslations,
    std::vector<pcl::Correspondences> &clustered_corrs,
    const float search_radius,
    const float hough_bin_size,
    const float hough_threshold) {
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr model_rf(
       new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf(
       new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::BOARDLocalReferenceFrameEstimation<
       PointT, pcl::Normal, pcl::ReferenceFrame> rf_est;
    rf_est.setFindHoles(true);
    rf_est.setRadiusSearch(search_radius);
    rf_est.setInputCloud(model_keypoints);
    rf_est.setInputNormals(model_normals);
    rf_est.setSearchSurface(model);
    rf_est.compute(*model_rf);
    
    rf_est.setInputCloud(scene_keypoints);
    rf_est.setInputNormals(scene_normals);
    rf_est.setSearchSurface(scene);
    rf_est.compute(*scene_rf);

    pcl::Hough3DGrouping<
       PointT, PointT, pcl::ReferenceFrame, pcl::ReferenceFrame> clusterer;
    clusterer.setHoughBinSize(hough_bin_size);
    clusterer.setHoughThreshold(hough_threshold);
    clusterer.setUseInterpolation(true);
    clusterer.setUseDistanceWeight(false);

    clusterer.setInputCloud(model_keypoints);
    clusterer.setInputRf(model_rf);
    clusterer.setSceneCloud(scene_keypoints);
    clusterer.setSceneRf(scene_rf);
    clusterer.setModelSceneCorrespondences(model_scene_corrs);
    clusterer.recognize(rototranslations, clustered_corrs);
}

int main(int argc, char *argv[]) {

    ros::init(argc, argv, "point_cloud_cluster_matching");
    PointCloudClusterMatching pccm;
    ros::spin();
    return 0;
}
