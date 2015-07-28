// Copyright (C) 2015 by Krishneel Chaudhary, JSK Lab,
// The University of Tokyo, Japan

#include <multilayer_object_tracking/multilayer_object_tracking.h>
#include <map>

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
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/multilayer_object_tracking/output/image", 1);

    this->pub_sindices_ = this->pnh_.advertise<
       jsk_recognition_msgs::ClusterPointIndices>(
          "/multilayer_object_tracking/supervoxel/indices", 1);
    this->pub_scloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
          "/multilayer_object_tracking/supervoxel/cloud", 1);
}

void MultilayerObjectTracking::subscribe() {
    this->sub_obj_cloud_.subscribe(this->pnh_, "input_obj_cloud", 1);
    this->sub_obj_pose_.subscribe(this->pnh_, "input_obj_pose", 1);
    this->obj_sync_ = boost::make_shared<message_filters::Synchronizer<
       ObjectSyncPolicy> >(100);
    this->obj_sync_->connectInput(sub_obj_cloud_, sub_obj_pose_);
    this->obj_sync_->registerCallback(
       boost::bind(&MultilayerObjectTracking::objInitCallback,
                   this, _1, _2));
    
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_pose_.subscribe(this->pnh_, "input_pose", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(sub_cloud_, sub_pose_);
    this->sync_->registerCallback(
       boost::bind(&MultilayerObjectTracking::callback,
                   this, _1, _2));
}

void MultilayerObjectTracking::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_pose_.unsubscribe();
    this->sub_obj_cloud_.unsubscribe();
    this->sub_obj_pose_.unsubscribe();
}

void MultilayerObjectTracking::objInitCallback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const geometry_msgs::PoseStamped::ConstPtr &pose_msg) {
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (this->init_counter_++ > 0) {
       ROS_WARN("Object is re-initalized! stopping & reseting...");
    }
    if (!cloud->empty()) {
       this->motion_history_.clear();
       PointXYZRPY motion_displacement;  // fix this
       this->estimatedPFPose(pose_msg, motion_displacement);
       
       std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;
       std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
       this->supervoxelSegmentation(cloud,
                                    supervoxel_clusters,
                                    supervoxel_adjacency);
       std::vector<AdjacentInfo> supervoxel_list;
       this->object_reference_ = ModelsPtr(new Models);
       this->processDecomposedCloud(
          cloud, supervoxel_clusters, supervoxel_adjacency,
          supervoxel_list, this->object_reference_);
    }
}

void MultilayerObjectTracking::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const geometry_msgs::PoseStamped::ConstPtr &pose_msg) {
    if (this->object_reference_->empty()) {
       ROS_WARN("No Model To Track Selected");
       return;
    }
    std::cout << "RUNNING CALLBACK---" << std::endl;
    
    // get PF pose of time t
    PointXYZRPY motion_displacement;
    this->estimatedPFPose(pose_msg, motion_displacement);
    std::cout << "Motion Displacement: " << motion_displacement << std::endl;
    
    // get the input cloud at time t
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    std::cout << "PROCESSING CLOUD....." << std::endl;
    this->globalLayerPointCloudProcessing(cloud, motion_displacement);
    std::cout << "CLOUD PROCESSED AND PUBLISHED" << std::endl;
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);

    ros_cloud_.header = cloud_msg->header;
    ros_indices_.header = cloud_msg->header;
    pub_scloud_.publish(ros_cloud_);
    pub_sindices_.publish(ros_indices_);
}

void MultilayerObjectTracking::processDecomposedCloud(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr> &
    supervoxel_clusters,
    const std::multimap<uint32_t, uint32_t> &supervoxel_adjacency,
    std::vector<AdjacentInfo> &supervoxel_list,
    MultilayerObjectTracking::ModelsPtr &models,
    bool norm_flag, bool feat_flag, bool cent_flag) {
    if (cloud->empty() || supervoxel_clusters.empty()) {
       return;
    }
    models = ModelsPtr(new Models);
    for (std::multimap<uint32_t, uint32_t>::const_iterator label_itr =
            supervoxel_adjacency.begin(); label_itr !=
            supervoxel_adjacency.end(); label_itr++) {
       ReferenceModel ref_model;
       ref_model.flag = true;
       uint32_t supervoxel_label = label_itr->first;
       pcl::Supervoxel<PointT>::Ptr supervoxel =
          supervoxel_clusters.at(supervoxel_label);
       if (supervoxel->voxels_->size() > min_cluster_size_) {
          std::vector<uint32_t> adjacent_voxels;
          for (std::multimap<uint32_t, uint32_t>::const_iterator
                  adjacent_itr = supervoxel_adjacency.equal_range(
                     supervoxel_label).first; adjacent_itr !=
                  supervoxel_adjacency.equal_range(
                     supervoxel_label).second; ++adjacent_itr) {
             pcl::Supervoxel<PointT>::Ptr neighbor_supervoxel =
                supervoxel_clusters.at(adjacent_itr->second);
             if (neighbor_supervoxel->voxels_->size() >
                 min_cluster_size_) {
                adjacent_voxels.push_back(adjacent_itr->second);
             }
          }
          AdjacentInfo a_info;
          a_info.adjacent_voxel_indices[supervoxel_label] =
             adjacent_voxels;
          supervoxel_list.push_back(a_info);
          a_info.voxel_index = supervoxel_label;
          ref_model.cluster_neigbors = a_info;
          ref_model.cluster_cloud = pcl::PointCloud<PointT>::Ptr(
             new pcl::PointCloud<PointT>);
          ref_model.cluster_cloud = supervoxel->voxels_;
          if (norm_flag) {
             ref_model.cluster_normals = pcl::PointCloud<pcl::Normal>::Ptr(
                new pcl::PointCloud<pcl::Normal>);
             ref_model.cluster_normals = supervoxel->normals_;
          }
          if (feat_flag) {
             this->computeCloudClusterRPYHistogram(
                ref_model.cluster_cloud,
                ref_model.cluster_normals,
                ref_model.cluster_vfh_hist);
             this->computeColorHistogram(
                ref_model.cluster_cloud,
                ref_model.cluster_color_hist);
          }
          if (cent_flag) {
             ref_model.cluster_centroid = Eigen::Vector4f();
             ref_model.cluster_centroid =
                supervoxel->centroid_.getVector4fMap();
          }
          ref_model.flag = false;
       }
       models->push_back(ref_model);
    }
}

void MultilayerObjectTracking::globalLayerPointCloudProcessing(
    pcl::PointCloud<PointT>::Ptr cloud,
    const MultilayerObjectTracking::PointXYZRPY &motion_disp) {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: Global Layer Input Empty");
       return;
    }
    pcl::PointCloud<PointT>::Ptr n_cloud(new pcl::PointCloud<PointT>);
    Models obj_ref = *object_reference_;
    std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;
    std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
    this->supervoxelSegmentation(cloud,
                                 supervoxel_clusters,
                                 supervoxel_adjacency);
    // TODO(remove below): REF: 2304893
    std::vector<AdjacentInfo> supervoxel_list;
    ModelsPtr t_voxels = ModelsPtr(new Models);
    this->processDecomposedCloud(
       cloud, supervoxel_clusters, supervoxel_adjacency,
       supervoxel_list, t_voxels, true, false, true);
    Models target_voxels = *t_voxels;
    std::map<int, int> matching_indices;
    // increase to neigbours neigbour incase of poor result
    for (int j = 0; j < obj_ref.size(); j++) {
       if (!obj_ref[j].flag) {
          float distance = FLT_MAX;
          int nearest_index = -1;
          Eigen::Vector4f obj_centroid;
          obj_centroid(0) = obj_ref[j].cluster_centroid(0) + motion_disp.x;
          obj_centroid(1) = obj_ref[j].cluster_centroid(1) + motion_disp.y;
          obj_centroid(2) = obj_ref[j].cluster_centroid(2) + motion_disp.z;
          obj_centroid(3) = 0.0f;
          for (int i = 0; i < target_voxels.size(); i++) {
             if (!target_voxels[i].flag) {
                Eigen::Vector4f t_centroid =
                   target_voxels[i].cluster_centroid;
                t_centroid(3) = 0.0f;
                float dist = static_cast<float>(
                   pcl::distances::l2(obj_centroid, t_centroid));
                if (dist < distance) {
                   distance = dist;
                   nearest_index = i;  // voxel_index
                }
             }
          }
          if (nearest_index != -1) {
             matching_indices[j] = nearest_index;
          }
       }
    }
    // set of patches that match the trajectory
    std::vector<uint32_t> best_match_index;
    pcl::PointCloud<PointT>::Ptr output(new pcl::PointCloud<PointT>);
    for (std::map<int, int>::iterator itr = matching_indices.begin();
         itr != matching_indices.end(); itr++) {
       if (!target_voxels[itr->second].flag) {
          std::map<uint32_t, std::vector<uint32_t> > neigb =
             target_voxels[itr->second].cluster_neigbors.adjacent_voxel_indices;
          uint32_t v_ind = target_voxels[
             itr->second].cluster_neigbors.voxel_index;
          best_match_index.push_back(v_ind);
          float probability = 0.0f;
          probability = this->targetCandidateToReferenceLikelihood<float>(
             obj_ref[itr->first], target_voxels[itr->second].cluster_cloud,
             target_voxels[itr->second].cluster_normals,
             target_voxels[itr->second].cluster_centroid);
          
          pcl::PointCloud<PointT>::Ptr match_cloud(new pcl::PointCloud<PointT>);
          *match_cloud = *target_voxels[itr->second].cluster_cloud;
          
          for (std::vector<uint32_t>::iterator it =
                  neigb.find(v_ind)->second.begin();
               it != neigb.find(v_ind)->second.end(); it++) {
             float prob = this->targetCandidateToReferenceLikelihood<float>(
                obj_ref[itr->first], supervoxel_clusters.at(*it)->voxels_,
                supervoxel_clusters.at(*it)->normals_,
                supervoxel_clusters.at(*it)->centroid_.getVector4fMap());
             if (prob > probability) {
                probability = prob;
                best_match_index.push_back(*it);

                match_cloud->clear();
                *match_cloud = *supervoxel_clusters.at(*it)->voxels_;
             }
             // *output = *output + *supervoxel_clusters.at(*it)->voxels_;
          }
          // best match local convexity


          // plot the best match
          // *output = *output + *supervoxel_clusters.at(v_ind)->voxels_;
          *output = *output + *match_cloud;
          // std::cout << "Probability: " << probability << std::endl;
       }
    }
    // get the neigbours of best match index
    for (std::vector<uint32_t>::iterator it = best_match_index.begin();
         it != best_match_index.end(); it++) {
       std::pair<std::multimap<uint32_t, uint32_t>::iterator,
                 std::multimap<uint32_t, uint32_t>::iterator> ret;
       ret = supervoxel_adjacency.equal_range(*it);
       Eigen::Vector4f c_centroid = supervoxel_clusters.at(
          *it)->centroid_.getVector4fMap();
       Eigen::Vector4f c_normal = this->cloudMeanNormal(
          supervoxel_clusters.at(*it)->normals_);
       *output = *output + *supervoxel_clusters.at(*it)->voxels_;
       for (std::multimap<uint32_t, uint32_t>::iterator itr = ret.first;
            itr != ret.second; itr++) {
          if (!supervoxel_clusters.at(itr->second)->voxels_->empty()) {
             Eigen::Vector4f n_centroid = supervoxel_clusters.at(
                itr->second)->centroid_.getVector4fMap();
             Eigen::Vector4f n_normal = this->cloudMeanNormal(
                supervoxel_clusters.at(itr->second)->normals_);
             float convx_weight = this->localVoxelConvexityLikelihood<float>(
                c_centroid, c_normal, n_centroid, n_normal);
             // std::cout << "Convex Region Prob: "
             // << convx_weight << std::endl;

             if (convx_weight != 0.0f) {
                *output = *output + *supervoxel_clusters.at(
                   itr->second)->voxels_;
             }
          }
          // *output = *output + *supervoxel_clusters.at(
          //    itr->second)->voxels_;
       }
    }
    cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*output, *cloud);
    this->publishSupervoxel(
       supervoxel_clusters, ros_cloud_, ros_indices_);
}

template<class T>
T MultilayerObjectTracking::targetCandidateToReferenceLikelihood(
    const MultilayerObjectTracking::ReferenceModel &reference_model,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normal,
    const Eigen::Vector4f &centroid) {
    if (cloud->empty() || normal->empty()) {
      return 0.0f;
    }
    cv::Mat vfh_hist;
    this->computeCloudClusterRPYHistogram(cloud, normal, vfh_hist);
    cv::Mat color_hist;
    this->computeColorHistogram(cloud, color_hist);
    T dist_vfh = static_cast<T>(
       cv::compareHist(vfh_hist,
                       reference_model.cluster_vfh_hist,
                       CV_COMP_BHATTACHARYYA));
    T dist_col = static_cast<T>(
       cv::compareHist(color_hist,
                       reference_model.cluster_color_hist,
                       CV_COMP_BHATTACHARYYA));
    T probability = std::exp(-1 * dist_vfh) * std::exp(-1 * dist_col);
    return probability;
}

template<class T>
T MultilayerObjectTracking::localVoxelConvexityLikelihood(
    Eigen::Vector4f c_centroid,
    Eigen::Vector4f c_normal,
    Eigen::Vector4f n_centroid,
    Eigen::Vector4f n_normal) {
    c_centroid(3) = 0.0f;
    c_normal(3) = 0.0f;
    n_centroid(3) = 0.0f;
    n_normal(3) = 0.0f;
    T weight = 1.0f;
    /*
    Eigen::Vector4f diff_vector = (c_centroid - n_centroid) / (
       c_centroid - n_centroid).norm();
    T connection = c_normal.dot(diff_vector) - n_normal.dot(diff_vector);
    if (connection > 0.0f) {
       weight = static_cast<T>(std::pow(1 - (c_normal.dot(n_normal)), 2));
    } else {
       return 0.0f;
    }
    */

    if ((n_centroid - c_centroid).dot(n_normal) > 0) {
       weight = static_cast<T>(std::pow(1 - (c_normal.dot(n_normal)), 2));
    } else {
       // weight = static_cast<T>(1 - (c_normal.dot(n_normal)));
       return 0.0f;
    }
    if (isnan(weight)) {
       return 0.0f;
    }
    
    T probability = std::exp(-1 * weight);
    return probability;
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
    cv::Mat &histogram) const {
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
    cv::Mat &hist, const int hBin, const int sBin, bool is_norm) const {
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

std::vector<MultilayerObjectTracking::AdjacentInfo>
MultilayerObjectTracking::voxelAdjacencyList(
    const jsk_recognition_msgs::AdjacencyList &adjacency_list) {
    std::vector<AdjacentInfo> supervoxel_list;
    AdjacentInfo tmp_list;
    for (int i = 0; i < adjacency_list.vertices.size(); i++) {
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
       if (!this->motion_history_.empty()) {
          int last_index = static_cast<int>(
             this->motion_history_.size()) - 1;
          motion_displacement.x = current_pose.x -
             this->motion_history_[last_index].x;
          motion_displacement.y = current_pose.y -
             this->motion_history_[last_index].y;
          motion_displacement.z = current_pose.z -
             this->motion_history_[last_index].z;
       } else {
          this->motion_history_.push_back(current_pose);
       }
    } else {
       // pertubate with history error weight
    }
}

void MultilayerObjectTracking::compute3DCentroids(
    const pcl::PointCloud<PointT>::Ptr cloud,
    Eigen::Vector4f &centre) const {
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
