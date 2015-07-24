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
       "/multilayer_object_tracking/output/cloud", 1);
    this->pub_indices_ = this->pnh_.advertise<
       jsk_recognition_msgs::ClusterPointIndices>(
       "/multilayer_object_tracking/output/indices", 1);
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/multilayer_object_tracking/output/image", 1);
}

void MultilayerObjectTracking::subscribe() {
    this->sub_obj_cloud_.subscribe(this->pnh_, "input_obj_cloud", 1);
    this->sub_obj_indices_.subscribe(this->pnh_, "input_obj_indices", 1);
    this->sub_obj_adj_.subscribe(this->pnh_, "input_obj_adj", 1);
    this->obj_sync_ = boost::make_shared<message_filters::Synchronizer<
       ObjectSyncPolicy> >(100);
    this->obj_sync_->connectInput(
       sub_obj_indices_, sub_obj_cloud_, sub_obj_adj_);
    this->obj_sync_->registerCallback(
       boost::bind(&MultilayerObjectTracking::objInitCallback,
                   this, _1, _2, _3));
       
    this->sub_indices_.subscribe(this->pnh_, "input_indices", 1);
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_pose_.subscribe(this->pnh_, "input_pose", 1);
    this->sub_adj_.subscribe(this->pnh_, "input_adj", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(sub_indices_, sub_cloud_, sub_adj_, sub_pose_);
    this->sync_->registerCallback(
       boost::bind(&MultilayerObjectTracking::callback,
                   this, _1, _2, _3, _4));
}

void MultilayerObjectTracking::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_indices_.unsubscribe();
}

void MultilayerObjectTracking::objInitCallback(
    const jsk_recognition_msgs::ClusterPointIndicesConstPtr &indices_mgs,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const jsk_recognition_msgs::AdjacencyList::ConstPtr &vertices_msg) {
    std::vector<pcl::PointIndices::Ptr> cluster_indices;
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    cluster_indices = this->clusterPointIndicesToPointIndices(indices_mgs);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (this->init_counter_++ > 0) {
       // reset tracking params
       ROS_WARN("Object is re-initalized! stopping & reseting...");
    }
    if (!cloud->empty()) {
       jsk_recognition_msgs::AdjacencyList adjacency_list = *vertices_msg;
       std::vector<pcl::PointIndices::Ptr> all_indices =
          this->clusterPointIndicesToPointIndices(indices_mgs);
       std::vector<AdjacentInfo> supervoxel_list =
          this->voxelAdjacencyList(adjacency_list);
       this->object_reference_ = ModelsPtr(new Models);
       this->processDecomposedCloud(
          cloud, all_indices, supervoxel_list, this->object_reference_);       
       Eigen::Vector4f centroid;
       this->compute3DCentroids(cloud, centroid);
       PointXYZRPY cur_position;
       cur_position.x = centroid(0);
       cur_position.y = centroid(1);
       cur_position.z = centroid(2);
       this->motion_history_.push_back(cur_position);
    }
}

void MultilayerObjectTracking::callback(
    const jsk_recognition_msgs::ClusterPointIndicesConstPtr &indices_mgs,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const jsk_recognition_msgs::AdjacencyList::ConstPtr &vertices_msg,
    const geometry_msgs::PoseStamped::ConstPtr &pose_msg) {
    if (this->object_reference_->empty()) {
       ROS_WARN("No Model To Track Selected");
       return;
    }

    std::cout << "RUNNING CALLBACK---" << std::endl;
    
    // get the indices of time t
    std::vector<pcl::PointIndices::Ptr> all_indices;
    all_indices = this->clusterPointIndicesToPointIndices(indices_mgs);

    // get PF pose of time t
    PointXYZRPY motion_displacement;
    this->estimatedPFPose(pose_msg, motion_displacement);
    std::cout << "Pose: " << motion_displacement << std::endl;

    // get the voxel connectivity at time t
    std::vector<AdjacentInfo> supervoxel_list =
       this->voxelAdjacencyList(*vertices_msg);
    
    // get the input cloud at time t
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    this->globalLayerPointCloudProcessing(cloud, supervoxel_list, all_indices);

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

void MultilayerObjectTracking::processDecomposedCloud(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const std::vector<pcl::PointIndices::Ptr> &all_indices,
    const std::vector<AdjacentInfo> &supervoxel_list,
    MultilayerObjectTracking::ModelsPtr &models) {
    if (all_indices.empty()) {
       return;
    }
    int icounter = 0;
    models = ModelsPtr(new Models);
    for (std::vector<pcl::PointIndices::Ptr>::const_iterator it =
            all_indices.begin(); it != all_indices.end(); it++) {
       ReferenceModel ref_model;
       ref_model.flag = true;
       if ((*it)->indices.size() > this->min_cluster_size_) {
          ref_model.cluster_cloud = pcl::PointCloud<PointT>::Ptr(
             new pcl::PointCloud<PointT>);
          ref_model.cluster_normals = pcl::PointCloud<pcl::Normal>::Ptr(
             new pcl::PointCloud<pcl::Normal>);
          pcl::ExtractIndices<PointT>::Ptr eifilter(
             new pcl::ExtractIndices<PointT>);
          eifilter->setInputCloud(cloud);
          eifilter->setIndices(*it);
          eifilter->filter(*ref_model.cluster_cloud);
          this->estimatePointCloudNormals<float>(
             ref_model.cluster_cloud,
             ref_model.cluster_normals,
             this->radius_search_);
          this->computeCloudClusterRPYHistogram(
             ref_model.cluster_cloud,
             ref_model.cluster_normals,
             ref_model.cluster_vfh_hist);
          this->computeColorHistogram(
             ref_model.cluster_cloud,
             ref_model.cluster_color_hist);
          ref_model.cluster_neigbors = supervoxel_list[icounter++];
          this->compute3DCentroids(ref_model.cluster_cloud,
                                   ref_model.cluster_centroid);
          ref_model.flag = false;
          
          // std::cout << "DEBUG: Model Info: "
          //           << ref_model.cluster_cloud->size() << "\t"
          //           << ref_model.cluster_normals->size() << "\t"
          //           << ref_model.cluster_vfh_hist.size() << "\t"
          //           << ref_model.cluster_color_hist.size()
          //           << std::endl;
       }
       models->push_back(ref_model);
    }
}

void MultilayerObjectTracking::globalLayerPointCloudProcessing(
    pcl::PointCloud<PointT>::Ptr cloud,
    const std::vector<AdjacentInfo> &supervoxel_list,
    const std::vector<pcl::PointIndices::Ptr> &all_indices) {
    if (cloud->empty() || all_indices.empty()) {
       ROS_ERROR("ERROR: Global Layer Input Empty");
       return;
    }
    pcl::PointCloud<PointT>::Ptr n_cloud(new pcl::PointCloud<PointT>);
    Models obj_ref = *object_reference_;
    ModelsPtr s_models = ModelsPtr(new Models);
    this->processDecomposedCloud(
       cloud, all_indices, supervoxel_list, s_models);
    Models scene_models = *s_models;
    for (int j = 0; j < scene_models.size(); j++) {
       ReferenceModel scene_model = scene_models[j];
       if (!scene_model.flag) {
          float probability = 0.0;
          for (int i = 0; i < obj_ref.size(); i++) {
             if (!obj_ref[i].flag) {
                // std::cout << i << "\tSize: "
                //           << obj_ref[i].cluster_vfh_hist.size()
                //           << obj_ref[i].cluster_color_hist.size()
                //           << scene_model.cluster_vfh_hist.size()
                //           << scene_model.cluster_color_hist.size()
                // << std::endl;
                
                float dist_vfh = static_cast<float>(
                   cv::compareHist(scene_model.cluster_vfh_hist,
                                   obj_ref[i].cluster_vfh_hist,
                                   CV_COMP_BHATTACHARYYA));
                float dist_col = static_cast<float>(
                   cv::compareHist(scene_model.cluster_color_hist,
                                   obj_ref[i].cluster_color_hist,
                                   CV_COMP_BHATTACHARYYA));
                
                // voxel neigbor weight
                // if works than move out to process once
                float obj_dist_weight;
                float obj_angle_weight;
                this->adjacentVoxelCoherencey(
                obj_ref, i, obj_dist_weight, obj_angle_weight);
                
                float s_dist_weight;
                float s_angle_weight;
                this->adjacentVoxelCoherencey(
                   scene_models, j, s_dist_weight, s_angle_weight);
                
                float prob = std::exp(-0.7 * dist_vfh) *
                   std::exp(-1.5 * dist_col);
                if (prob > probability /*&& prob > 0.5f*/) {
                   probability = prob;
                }
             }
          }
          std::cout << "Probability: " << probability << std::endl;
          for (int i = 0; i < scene_model.cluster_cloud->size(); i++) {
             PointT pt = scene_model.cluster_cloud->points[i];
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

std::vector<MultilayerObjectTracking::AdjacentInfo>
MultilayerObjectTracking::voxelAdjacencyList(
    const jsk_recognition_msgs::AdjacencyList &adjacency_list) {
    std::vector<AdjacentInfo> supervoxel_list;
    AdjacentInfo tmp_list;
    for (int i = 0 ; i < adjacency_list.vertices.size(); i++) {
       int vertex_index = adjacency_list.vertices[i];
       float dist = static_cast<float>(adjacency_list.edge_weight[i]);
       if (vertex_index == -1) {
          supervoxel_list.push_back(tmp_list);
          tmp_list.adjacent_indices.clear();
          tmp_list.adjacent_distances.clear();
       } else {
          tmp_list.adjacent_indices.push_back(vertex_index);
          tmp_list.adjacent_distances.push_back(dist);
       }
    }
    return supervoxel_list;
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

void MultilayerObjectTracking::estimatedPFPose(
    const geometry_msgs::PoseStamped::ConstPtr &pose_msg,
    PointXYZRPY &motion_displacement) {
    PointXYZRPY current_pose;
    current_pose.x = pose_msg->pose.position.x;
    current_pose.y = pose_msg->pose.position.y;
    current_pose.z = pose_msg->pose.position.z;
    current_pose.roll = pose_msg->pose.orientation.x;
    current_pose.pitch = pose_msg->pose.orientation.y;
    current_pose.yaw = pose_msg->pose.orientation.z;
    current_pose.weight = pose_msg->pose.orientation.w;
    if (!isnan(current_pose.x) && !isnan(current_pose.y) &&
        !isnan(current_pose.z)) {
       int last_index = static_cast<int>(this->motion_history_.size()) - 1;
       motion_displacement.x = current_pose.x -
          this->motion_history_[last_index].x;
       motion_displacement.y = current_pose.y -
          this->motion_history_[last_index].y;
       motion_displacement.z = current_pose.z -
          this->motion_history_[last_index].z;
       this->motion_history_.push_back(current_pose);
    } else {
       // pertubate with history error weight
    }
}

void MultilayerObjectTracking::compute3DCentroids(
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
    }
}

Eigen::Vector4f MultilayerObjectTracking::cloudMeanNormal(
    const pcl::PointCloud<pcl::Normal>::Ptr normal,
    bool isnorm) {
    if (normal->empty()) {
       return Eigen::Vector4f(0, 0, 0, 0);
    }
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    int icounter = 0;
    for (int i = 0; i < normal->size(); i++) {
       if ((!isnan(normal->points[i].normal_x)) &&
           (!isnan(normal->points[i].normal_y)) &&
           (!isnan(normal->points[i].normal_z))) {
          x += normal->points[i].normal_x;
          y += normal->points[i].normal_y;
          z += normal->points[i].normal_z;
          icounter++;
       }
    }
    Eigen::Vector4f n_mean = Eigen::Vector4f(
       x/static_cast<float>(icounter),
       y/static_cast<float>(icounter),
       z/static_cast<float>(icounter),
       0.0f);
    if (isnorm) {
       n_mean.normalize();
    }
    return n_mean;
}

float MultilayerObjectTracking::computeCoherency(
    const float dist, const float weight) {
    if (isnan(dist)) {
       return 0.0f;
    }
    return static_cast<float>(1/(1 + (weight * std::pow(dist, 2))));
}

void MultilayerObjectTracking::adjacentVoxelCoherencey(
    const Models &ref_model, const int index,
    float &dist_weight, float &angle_weight) {
    ReferenceModel object_model = ref_model[index];
    if (object_model.flag) {
       return;
    }
    AdjacentInfo adjacent_info = object_model.cluster_neigbors;
    dist_weight = 0.0f;
    angle_weight = 0.0f;
    Eigen::Vector4f c_mean = this->cloudMeanNormal(
       object_model.cluster_normals);
    int icounter = 0;
    for (int i = 1; i < adjacent_info.adjacent_indices.size(); i++) {
       int nidx = adjacent_info.adjacent_indices[i] - 1;
       if (!ref_model[nidx].flag && nidx < ref_model.size()) {
          Eigen::Vector4f n_mean = this->cloudMeanNormal(
             ref_model[nidx].cluster_normals);
          float dist = adjacent_info.adjacent_distances[i];
          dist_weight += this->computeCoherency(dist, 1.0f);
          float theta = static_cast<float>(pcl::getAngle3D(c_mean, n_mean));
          angle_weight += this->computeCoherency(theta, 1.0f);
          icounter++;
       }
    }
    dist_weight /= static_cast<float>(icounter);
    angle_weight /= static_cast<float>(icounter);
}

int main(int argc, char *argv[]) {

    ros::init(argc, argv, "multilayer_object_tracking");
    MultilayerObjectTracking mot;
    ros::spin();
    return 0;
}
