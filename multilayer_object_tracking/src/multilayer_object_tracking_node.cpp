// Copyright (C) 2015 by Krishneel Chaudhary, JSK Lab,
// The University of Tokyo, Japan

#include <multilayer_object_tracking/multilayer_object_tracking.h>
#include <map>

MultilayerObjectTracking::MultilayerObjectTracking() :
    init_counter_(0) {
    this->object_reference_ = ModelsPtr(new Models);
    this->clustering_client_ = this->pnh_.serviceClient<
       multilayer_object_tracking::EstimatedCentroidsClustering>(
          "estimated_centroids_clustering");
    this->onInit();
}

void MultilayerObjectTracking::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/multilayer_object_tracking/output/cloud", 1);

    this->pub_sindices_ = this->pnh_.advertise<
       jsk_recognition_msgs::ClusterPointIndices>(
          "/multilayer_object_tracking/supervoxel/indices", 1);
    this->pub_scloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
          "/multilayer_object_tracking/supervoxel/cloud", 1);

    this->pub_normal_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
        "/multilayer_object_tracking/output/normal", sizeof(char));

    this->pub_tdp_ = this->pnh_.advertise<
       jsk_recognition_msgs::ClusterPointIndices>(
          "/multilayer_object_tracking/supervoxel/tdp_indices", 1);

    this->pub_inliers_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/multilayer_object_tracking/output/inliers", 1);

    this->pub_centroids_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
        "/multilayer_object_tracking/output/centroids", 1);
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
          supervoxel_list, this->object_reference_, true, true, true, true);
    }
}

void MultilayerObjectTracking::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const geometry_msgs::PoseStamped::ConstPtr &pose_msg) {
    if (this->object_reference_->empty()) {
       ROS_WARN("No Model To Track Selected");
       return;
    }
    ROS_INFO("\n\n--------------RUNNING CALLBACK-------------------");
    ros::Time begin = ros::Time::now();
    
    // get PF pose of time t
    PointXYZRPY motion_displacement;
    this->estimatedPFPose(pose_msg, motion_displacement);
    std::cout << "Motion Displacement: " << motion_displacement << std::endl;
    
    // get the input cloud at time t
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    ROS_INFO("PROCESSING CLOUD.....");
    this->globalLayerPointCloudProcessing(
       cloud, motion_displacement, cloud_msg->header);
    ROS_INFO("CLOUD PROCESSED AND PUBLISHED");

    ros::Time end = ros::Time::now();
    std::cout << "Processing Time: " << end - begin << std::endl;
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

void MultilayerObjectTracking::processDecomposedCloud(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr> &
    supervoxel_clusters,
    const std::multimap<uint32_t, uint32_t> &supervoxel_adjacency,
    std::vector<AdjacentInfo> &supervoxel_list,
    MultilayerObjectTracking::ModelsPtr &models,
    bool norm_flag, bool feat_flag, bool cent_flag, bool neigh_pfh) {
    if (cloud->empty() || supervoxel_clusters.empty()) {
       return;
    }
    models = ModelsPtr(new Models);
    int icounter = 0;
    for (std::multimap<uint32_t, pcl::Supervoxel<PointT>::Ptr>::const_iterator
            label_itr = supervoxel_clusters.begin(); label_itr !=
            supervoxel_clusters.end(); label_itr++) {
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
             icounter++;
          }
          AdjacentInfo a_info;
          a_info.adjacent_voxel_indices[supervoxel_label] = adjacent_voxels;
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
          models->push_back(ref_model);
       }
    }
    std::cout << "Model size: "  << models->size() << std::endl;

    // compute the local pfh
    if (neigh_pfh) {
       for (int i = 0; i < models->size(); i++) {
            this->computeLocalPairwiseFeautures(
                supervoxel_clusters,
                models->operator[](i).cluster_neigbors.adjacent_voxel_indices,
                models->operator[](i).neigbour_pfh);
        }
    }
}

void MultilayerObjectTracking::globalLayerPointCloudProcessing(
    pcl::PointCloud<PointT>::Ptr cloud,
    const MultilayerObjectTracking::PointXYZRPY &motion_disp,
    const std_msgs::Header header) {
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
    Eigen::Matrix<float, 3, 3> rotation_matrix;
    this->getRotationMatrixFromRPY<float>(motion_disp, rotation_matrix);
    
    // TODO(remove below): REF: 2304893
    std::vector<AdjacentInfo> supervoxel_list;
    ModelsPtr t_voxels = ModelsPtr(new Models);
    this->processDecomposedCloud(
       cloud, supervoxel_clusters, supervoxel_adjacency,
       supervoxel_list, t_voxels, true, false, true);
    Models target_voxels = *t_voxels;
    std::map<int, int> matching_indices;  // hold the query and test case
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
             // TODO(.): >>> move next loop here <<<<
          }
          // computing the model object cluster centroid-centroid ratio
          const int motion_hist_index = this->motion_history_.size() - 1;
          obj_ref[j].centroid_distance(0) = obj_ref[j].cluster_centroid(0) -
             this->motion_history_[motion_hist_index].x;
          obj_ref[j].centroid_distance(1) = obj_ref[j].cluster_centroid(1) -
             this->motion_history_[motion_hist_index].y;
          obj_ref[j].centroid_distance(2) = obj_ref[j].cluster_centroid(2) -
             this->motion_history_[motion_hist_index].z;
       }
    }
    // NOTE: if the VFH matches are on the BG than perfrom
    // backprojection to confirm the match thru motion and VFH
    // set of patches that match the trajectory
    int counter = 0;
    pcl::PointCloud<PointT>::Ptr est_centroid_cloud(
       new pcl::PointCloud<PointT>);
    std::multimap<uint32_t, Eigen::Vector3f> estimated_centroids;
    std::vector<uint32_t> best_match_index;
    for (std::map<int, int>::iterator itr = matching_indices.begin();
         itr != matching_indices.end(); itr++) {
       if (!target_voxels[itr->second].flag) {
          std::map<uint32_t, std::vector<uint32_t> > neigb =
             target_voxels[itr->second].cluster_neigbors.adjacent_voxel_indices;
          uint32_t v_ind = target_voxels[
             itr->second].cluster_neigbors.voxel_index;
          uint32_t bm_index = v_ind;
          float probability = 0.0f;
          probability = this->targetCandidateToReferenceLikelihood<float>(
             obj_ref[itr->first], target_voxels[itr->second].cluster_cloud,
             target_voxels[itr->second].cluster_normals,
             target_voxels[itr->second].cluster_centroid);
          
          // TODO(.) collect the neigbours here instead of next for
          // loop
          
          cv::Mat histogram_phf;
          this->computeLocalPairwiseFeautures(
             supervoxel_clusters, neigb, histogram_phf);
          // ------------------------------------
          float local_weight = 0.0f;  // use to weight the center transformation
          for (std::vector<uint32_t>::iterator it =
                  neigb.find(v_ind)->second.begin();
               it != neigb.find(v_ind)->second.end(); it++) {
             float prob = this->targetCandidateToReferenceLikelihood<float>(
                obj_ref[itr->first], supervoxel_clusters.at(*it)->voxels_,
                supervoxel_clusters.at(*it)->normals_,
                supervoxel_clusters.at(*it)->centroid_.getVector4fMap());

             // ---- computation of local neigbour orientations-----
             std::map<uint32_t, std::vector<uint32_t> > local_adjacency;
             std::vector<uint32_t> list_adj;
             for (std::multimap<uint32_t, uint32_t>::const_iterator
                     adjacent_itr = supervoxel_adjacency.equal_range(
                        *it).first; adjacent_itr !=
                     supervoxel_adjacency.equal_range(*it).second;
                  ++adjacent_itr) {
                list_adj.push_back(adjacent_itr->second);
             }
             local_adjacency[*it] = list_adj;
             cv::Mat local_phf;
             this->computeLocalPairwiseFeautures(
                supervoxel_clusters, local_adjacency, local_phf);
             
             float dist_phf = static_cast<float>(
                cv::compareHist(obj_ref[itr->first].neigbour_pfh,
                                local_phf, CV_COMP_BHATTACHARYYA));
             float phf_prob = std::exp(-1 * dist_phf);
             // std::cout << phf_prob << "  ";
             // prob *= phf_prob;
             // -----------------------------------------------------

             if (prob > probability) {
                 probability = prob;
                bm_index = *it;
             }
             local_weight = phf_prob;
          }
          if (probability > threshold_) {
             std::cout << "Probability: " << local_weight << std::endl;
              best_match_index.push_back(bm_index);

              // voting for centroid
              Eigen::Vector3f estimated_position = supervoxel_clusters.at(
                 bm_index)->centroid_.getVector3fMap() - rotation_matrix *
                  obj_ref[itr->first].centroid_distance /* local_weight*/;
              estimated_centroids.insert(std::pair<
                 uint32_t, Eigen::Vector3f>(bm_index, estimated_position));
              
              PointT pt;
              pt.x = estimated_position(0);
              pt.y = estimated_position(1);
              pt.z = estimated_position(2);
              pt.r = 255;
              est_centroid_cloud->push_back(pt);
              counter++;
          }
       }
    }
    // centroid votes clustering
    pcl::PointCloud<PointT>::Ptr inliers(new pcl::PointCloud<PointT>);
    this->estimatedCentroidClustering(
        estimated_centroids, inliers, best_match_index);
    
    // -----
    std::cout << "TOTAL POINTS: " << estimated_centroids.size() << std::endl;
    std::cout << "Cloud Size: " << est_centroid_cloud->size() << "\t"
              << inliers->size() << "\t" <<
       counter << "\t Best Match: " << best_match_index.size() << std::endl;
    // -----
    
    // for visualization of normals on rviz
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr centroid_normal(
       new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    // get the neigbours of best match index
    pcl::PointCloud<PointT>::Ptr output(new pcl::PointCloud<PointT>);
    std::vector<uint32_t> neigb_lookup;
    neigb_lookup = best_match_index;   // copy the best set
    ModelsPtr update_ref_model(new Models);
    for (std::vector<uint32_t>::iterator it = best_match_index.begin();
         it != best_match_index.end(); it++) {
       std::pair<std::multimap<uint32_t, uint32_t>::iterator,
                 std::multimap<uint32_t, uint32_t>::iterator> ret;
       ret = supervoxel_adjacency.equal_range(*it);
       Eigen::Vector4f c_centroid = supervoxel_clusters.at(
          *it)->centroid_.getVector4fMap();
       Eigen::Vector4f c_normal = this->cloudMeanNormal(
          supervoxel_clusters.at(*it)->normals_);

       centroid_normal->push_back(
          this->convertVector4fToPointXyzRgbNormal(
             c_centroid, c_normal, cv::Scalar(255, 0, 0)));

       // update the reference models with direct match voxels
       ReferenceModel ref_model;
       this->processVoxelForReferenceModel(supervoxel_clusters,
           supervoxel_adjacency, *it, ref_model);
       update_ref_model->push_back(ref_model);
       
       // neigbour voxel convex relationship
       for (std::multimap<uint32_t, uint32_t>::iterator itr = ret.first;
            itr != ret.second; itr++) {
          bool is_process_neigh = true;
          for (std::vector<uint32_t>::iterator lup_it = neigb_lookup.begin();
               lup_it != neigb_lookup.end(); lup_it++) {
             if (*lup_it == itr->second) {
                is_process_neigh = false;
                break;
             }
          }
          if (!supervoxel_clusters.at(itr->second)->voxels_->empty() &&
              is_process_neigh) {
              /*
                neigb_lookup.push_back(itr->second);
             Eigen::Vector4f n_centroid = supervoxel_clusters.at(
                itr->second)->centroid_.getVector4fMap();
             Eigen::Vector4f n_normal = this->cloudMeanNormal(
                supervoxel_clusters.at(itr->second)->normals_);
             float convx_weight = this->localVoxelConvexityLikelihood<float>(
                c_centroid, c_normal, n_centroid, n_normal);
             if (convx_weight > 0.0f) {
                *output = *output + *supervoxel_clusters.at(
                   itr->second)->voxels_;
                centroid_normal->push_back(
                   this->convertVector4fToPointXyzRgbNormal(
                      n_centroid, n_normal, cv::Scalar(0, 255, 0)));
             }
              */
             // std::cout << convx_weight << "\t";
             // ------------------------------------------
             // get the common neigbor to both
             std::pair<std::multimap<uint32_t, uint32_t>::iterator,
                       std::multimap<uint32_t, uint32_t>::iterator> comm_neigb;
             comm_neigb = supervoxel_adjacency.equal_range(itr->second);
             uint32_t common_neigbour_index = 0;
             for (std::multimap<uint32_t, uint32_t>::iterator c_itr =
                     comm_neigb.first; c_itr != comm_neigb.second; c_itr++) {
                if (!supervoxel_clusters.at(c_itr->second)->voxels_->empty()) {
                   bool is_common_neigh = false;
                   for (std::map<uint32_t, uint32_t>::iterator itr_ret =
                           supervoxel_adjacency.equal_range(c_itr->first).
                           first; itr_ret != supervoxel_adjacency.equal_range(
                              c_itr->first).second; itr_ret++) {
                      if (itr_ret->second == *it) {
                         is_common_neigh = true;
                         common_neigbour_index = c_itr->second;  // check ?
                         break;
                      }
                   }
                   if (is_common_neigh) {
                      // std::cout << "Common Neigbor exists..." << std::endl;
                      break;
                   }
                }
             }
             if (common_neigbour_index > 0) {
                Eigen::Vector4f n_centroid_b = supervoxel_clusters.at(
                   itr->second)->centroid_.getVector4fMap();
                Eigen::Vector4f n_normal_b = this->cloudMeanNormal(
                   supervoxel_clusters.at(itr->second)->normals_);
                Eigen::Vector4f n_centroid_c = supervoxel_clusters.at(
                   common_neigbour_index)->centroid_.getVector4fMap();
                Eigen::Vector4f n_normal_c = this->cloudMeanNormal(
                   supervoxel_clusters.at(common_neigbour_index)->normals_);
                float convx_weight_ab = this->localVoxelConvexityLikelihood<
                   float>(c_centroid, c_normal, n_centroid_b, n_normal_b);
                float convx_weight_ac = this->localVoxelConvexityLikelihood<
                   float>(c_centroid, c_normal, n_centroid_c, n_normal_c);
                float convx_weight_bc = this->localVoxelConvexityLikelihood<
                   float>(n_centroid_b, n_normal_b, n_centroid_c, n_normal_c);
                 
                if (convx_weight_ab != 0.0f &&
                    convx_weight_ac != 0.0f &&
                    convx_weight_bc != 0.0f) {
                   *output = *output + *supervoxel_clusters.at(
                      itr->second)->voxels_;
                   centroid_normal->push_back(
                      this->convertVector4fToPointXyzRgbNormal(
                          n_centroid_b, n_normal_b, cv::Scalar(0, 255, 0)));
                   neigb_lookup.push_back(itr->second);
                   // add the surfels to the model (obj_ref)
                   
                   ReferenceModel ref_model;
                   this->processVoxelForReferenceModel(supervoxel_clusters,
                       supervoxel_adjacency, itr->second, ref_model);
                   update_ref_model->push_back(ref_model);
                   
                }
                // std::cout << convx_weight_ab << "\t";
             }
             // ------------------------------------------
          }
       }
       *output = *output + *supervoxel_clusters.at(*it)->voxels_;
    }
    if (update_ref_model->size() > 5) {
        this->object_reference_->clear();
        this->object_reference_ = update_ref_model;
    }
    
    cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*output, *cloud);
    
    pcl::PointIndices tdp_ind;
    for (int i = 0; i < cloud->size(); i++) {
       tdp_ind.indices.push_back(i);
    }

    /* visualization of target surfels */
    std::vector<pcl::PointIndices> all_indices;
    all_indices.push_back(tdp_ind);
    jsk_recognition_msgs::ClusterPointIndices tdp_indices;
    tdp_indices.cluster_indices = this->convertToROSPointIndices(
       all_indices, header);
    tdp_indices.header = header;
    this->pub_tdp_.publish(tdp_indices);
    
    /* for visualization of supervoxel */
    sensor_msgs::PointCloud2 ros_svcloud;
    jsk_recognition_msgs::ClusterPointIndices ros_svindices;
    this->publishSupervoxel(
       supervoxel_clusters, ros_svcloud, ros_svindices, header);
    pub_scloud_.publish(ros_svcloud);
    pub_sindices_.publish(ros_svindices);

    /* for visualization of inliers */
    sensor_msgs::PointCloud2 ros_inliers;
    pcl::toROSMsg(*inliers, ros_inliers);
    ros_inliers.header = header;
    this->pub_inliers_.publish(ros_inliers);

    /* for visualization of initial centroids */
    sensor_msgs::PointCloud2 ros_centroids;
    pcl::toROSMsg(*est_centroid_cloud, ros_centroids);
    ros_centroids.header = header;
    this->pub_centroids_.publish(ros_centroids);
    
    /* for visualization of normal */
    if (!centroid_normal->empty()) {
       sensor_msgs::PointCloud2 rviz_normal;
       pcl::toROSMsg(*centroid_normal, rviz_normal);
       rviz_normal.header = header;
       this->pub_normal_.publish(rviz_normal);
    }
}

void MultilayerObjectTracking::processVoxelForReferenceModel(
    const std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters,
    const std::multimap<uint32_t, uint32_t> supervoxel_adjacency,
    const uint32_t match_index,
    MultilayerObjectTracking::ReferenceModel &ref_model) {
    if (supervoxel_clusters.empty() || supervoxel_adjacency.empty()) {
        ROS_ERROR("ERROR: empty data for updating voxel ref model");
        return;
    }
    if (supervoxel_clusters.at(
            match_index)->voxels_->size() > this->min_cluster_size_) {
        ref_model.flag = false;
        ref_model.cluster_cloud = supervoxel_clusters.at(
            match_index)->voxels_;
        ref_model.cluster_normals = supervoxel_clusters.at(
            match_index)->normals_;
        ref_model.cluster_centroid = supervoxel_clusters.at(
            match_index)->centroid_.getVector4fMap();
        this->computeCloudClusterRPYHistogram(
            ref_model.cluster_cloud,
            ref_model.cluster_normals,
            ref_model.cluster_vfh_hist);
        this->computeColorHistogram(
            ref_model.cluster_cloud,
            ref_model.cluster_color_hist);
        std::vector<uint32_t> adjacent_voxels;
        for (std::multimap<uint32_t, uint32_t>::const_iterator adjacent_itr =
                 supervoxel_adjacency.equal_range(match_index).first;
             adjacent_itr != supervoxel_adjacency.equal_range(
                 match_index).second; ++adjacent_itr) {
            pcl::Supervoxel<PointT>::Ptr neighbor_supervoxel =
                supervoxel_clusters.at(adjacent_itr->second);
            if (neighbor_supervoxel->voxels_->size() >
                min_cluster_size_) {
                adjacent_voxels.push_back(adjacent_itr->second);
            }
        }
        AdjacentInfo a_info;
        a_info.adjacent_voxel_indices[match_index] = adjacent_voxels;
        a_info.voxel_index = match_index;
        ref_model.cluster_neigbors = a_info;
        std::map<uint32_t, std::vector<uint32_t> > local_adj;
        local_adj[match_index] = adjacent_voxels;
        this->computeLocalPairwiseFeautures(
            supervoxel_clusters, local_adj, ref_model.neigbour_pfh);
    }
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
    bool is_gfpfh = false;
    bool is_vfh = true;
    if (is_gfpfh) {
        pcl::PointCloud<pcl::PointXYZL>::Ptr object(
            new pcl::PointCloud<pcl::PointXYZL>);
        for (int i = 0; i < cloud->size(); i++) {
            pcl::PointXYZL pt;
            pt.x = cloud->points[i].x;
            pt.y = cloud->points[i].y;
            pt.z = cloud->points[i].z;
            pt.label = 1;
            object->push_back(pt);
        }
        pcl::GFPFHEstimation<
            pcl::PointXYZL, pcl::PointXYZL, pcl::GFPFHSignature16> gfpfh;
        gfpfh.setInputCloud(object);
        gfpfh.setInputLabels(object);
        gfpfh.setOctreeLeafSize(0.01);
        gfpfh.setNumberOfClasses(1);
        pcl::PointCloud<pcl::GFPFHSignature16>::Ptr descriptor(
            new pcl::PointCloud<pcl::GFPFHSignature16>);
        gfpfh.compute(*descriptor);
        histogram = cv::Mat(sizeof(char), 16, CV_32F);
        for (int i = 0; i < histogram.cols; i++) {
            histogram.at<float>(0, i) = descriptor->points[0].histogram[i];

        }
    }
    if (is_vfh) {
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
          motion_displacement.roll = current_pose.roll -
             this->motion_history_[last_index].roll;
          motion_displacement.pitch = current_pose.pitch -
              this->motion_history_[last_index].pitch;
          motion_displacement.yaw = current_pose.yaw -
              this->motion_history_[last_index].yaw;
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

void MultilayerObjectTracking::computeScatterMatrix(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const Eigen::Vector4f centroid) {
    if (cloud->empty()) {
       ROS_ERROR("Empty input for computing Scatter Matrix");
       return;
    }
    const int rows = 3;
    const int cols = 3;
    Eigen::MatrixXf scatter_matrix = Eigen::Matrix3f::Zero(cols, rows);
    for (int i = 0; i < cloud->size(); i++) {
       Eigen::Vector3f de_mean = Eigen::Vector3f();
       de_mean(0) = cloud->points[i].x - centroid(0);
       de_mean(1) = cloud->points[i].y - centroid(1);
       de_mean(2) = cloud->points[i].z - centroid(2);
       Eigen::Vector3f t_de_mean = de_mean.transpose();
       for (int y = 0; y < rows; y++) {
          for (int x = 0; x < cols; x++) {
             scatter_matrix(y, x) += de_mean(y) * t_de_mean(x);
          }
       }
    }
    Eigen::EigenSolver<Eigen::MatrixXf> eigen_solver(scatter_matrix, true);
    // Eigen::complex<float> eigen_values;
}

float MultilayerObjectTracking::computeCoherency(
    const float dist, const float weight) {
    if (isnan(dist)) {
       return 0.0f;
    }
    return static_cast<float>(1/(1 + (weight * std::pow(dist, 2))));
}

pcl::PointXYZRGBNormal
MultilayerObjectTracking::convertVector4fToPointXyzRgbNormal(
     const Eigen::Vector4f &centroid,
     const Eigen::Vector4f &normal,
     const cv::Scalar color) {
     pcl::PointXYZRGBNormal pt;
     pt.x = centroid(0);
     pt.y = centroid(1);
     pt.z = centroid(2);
     pt.r = color.val[2];
     pt.g = color.val[1];
     pt.b = color.val[0];
     pt.normal_x = normal(0);
     pt.normal_y = normal(1);
     pt.normal_z = normal(2);
     return pt;
}

template<typename T>
void MultilayerObjectTracking::getRotationMatrixFromRPY(
    const PointXYZRPY &motion_displacement,
    Eigen::Matrix<T, 3, 3> &rotation) {
    tf::Quaternion tf_quaternion;
    tf_quaternion.setEulerZYX(motion_displacement.yaw,
                              motion_displacement.pitch,
                              motion_displacement.roll);
    Eigen::Quaternion<float> quaternion = Eigen::Quaternion<float>(
        tf_quaternion.w(), tf_quaternion.x(),
        tf_quaternion.y(), tf_quaternion.z());
    rotation.template block<3, 3>(0, 0) =
        quaternion.normalized().toRotationMatrix();
}

void MultilayerObjectTracking::computeLocalPairwiseFeautures(
    const std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr> &
    supervoxel_clusters, const std::map<uint32_t, std::vector<uint32_t> > &
    adjacency_list, cv::Mat &histogram, const int feature_count) {
    if (supervoxel_clusters.empty() || adjacency_list.empty()) {
       std::cout << supervoxel_clusters.size()  << "\t"
                 << adjacency_list.size() << std::endl;
       ROS_ERROR("ERROR: empty data returing no local feautures");
       return;
    }
    float d_pi = 1.0f / (2.0f * static_cast<float> (M_PI));
    histogram = cv::Mat::zeros(1, this->bin_size_ * feature_count, CV_32F);
    for (std::map<uint32_t, std::vector<uint32_t> >::const_iterator it =
            adjacency_list.begin(); it != adjacency_list.end(); it++) {
       pcl::Supervoxel<PointT>::Ptr supervoxel =
          supervoxel_clusters.at(it->first);
       Eigen::Vector4f c_normal = this->cloudMeanNormal(supervoxel->normals_);
       std::map<uint32_t, Eigen::Vector4f> cache_normals;
       int icounter = 0;
       for (std::vector<uint32_t>::const_iterator itr = it->second.begin();
            itr != it->second.end(); itr++) {
          pcl::Supervoxel<PointT>::Ptr neighbor_supervoxel =
             supervoxel_clusters.at(*itr);
          Eigen::Vector4f n_normal = this->cloudMeanNormal(
             neighbor_supervoxel->normals_);
          float alpha;
          float phi;
          float theta;
          float distance;
          pcl::computePairFeatures(
             supervoxel->centroid_.getVector4fMap(), c_normal,
             neighbor_supervoxel->centroid_.getVector4fMap(),
             n_normal, alpha, phi, theta, distance);
          int bin_index[feature_count];
          bin_index[0] = static_cast<int>(
             std::floor(this->bin_size_ * ((alpha + M_PI) * d_pi)));
          bin_index[1] = static_cast<int>(
             std::floor(this->bin_size_ * ((phi + 1.0f) * 0.5f)));
          bin_index[2] = static_cast<int>(
             std::floor(this->bin_size_ * ((theta + 1.0f) * 0.5f)));
          for (int i = 0; i < feature_count; i++) {
             if (bin_index[i] < 0) {
                bin_index[i] = 0;
             }
             if (bin_index[i] >= bin_size_) {
                bin_index[i] = bin_size_ - sizeof(char);
             }
             // h_index += h_p + bin_index[i];
             // h_p *= this->bin_size_;
             histogram.at<float>(
                0, bin_index[i] + (i * this->bin_size_)) += 1;
          }
          cache_normals[*itr] = n_normal;
          icounter++;
       }
       // std::cout << "Total Neigbours" << icounter << std::endl;
       // neigbour-neigbour phf
       for (std::vector<uint32_t>::const_iterator itr = it->second.begin();
            itr != it->second.end(); itr++) {
          cache_normals.find(*itr)->second;
          for (std::vector<uint32_t>::const_iterator itr_in =
                  it->second.begin(); itr_in != it->second.end(); itr_in++) {
             if (*itr != *itr_in) {
                float alpha;
                float phi;
                float theta;
                float distance;
                pcl::computePairFeatures(
                   supervoxel_clusters.at(*itr)->centroid_.getVector4fMap(),
                   cache_normals.find(*itr)->second, supervoxel_clusters.at(
                      *itr_in)->centroid_.getVector4fMap(),
                   cache_normals.find(*itr_in)->second,
                   alpha, phi, theta, distance);
                int bin_index[feature_count];
                bin_index[0] = static_cast<int>(
                   std::floor(this->bin_size_ * ((alpha + M_PI) * d_pi)));
                bin_index[1] = static_cast<int>(
                   std::floor(this->bin_size_ * ((phi + 1.0f) * 0.5f)));
                bin_index[2] = static_cast<int>(
                   std::floor(this->bin_size_ * ((theta + 1.0f) * 0.5f)));
                for (int i = 0; i < feature_count; i++) {
                   if (bin_index[i] < 0) {
                      bin_index[i] = 0;
                   }
                   if (bin_index[i] >= bin_size_) {
                      bin_index[i] = bin_size_ - sizeof(char);
                   }
                   histogram.at<float>(
                      0, bin_index[i] + (i * this->bin_size_)) += 1;
                }
             }
          }
       }
       cv::normalize(
          histogram, histogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    }
}

void MultilayerObjectTracking::estimatedCentroidClustering(
    const std::multimap<uint32_t, Eigen::Vector3f> &estimated_centroids,
    pcl::PointCloud<PointT>::Ptr inliers,
    std::vector<uint32_t> &best_match_index) {
    if (estimated_centroids.size() < this->eps_min_samples_ + sizeof(char)) {
        ROS_WARN("Too Little Points for Clustering\n ...Skipping...\n");
        return;
    }
    multilayer_object_tracking::EstimatedCentroidsClustering ecc_srv;
    for (std::map<uint32_t, Eigen::Vector3f>::const_iterator it =
             estimated_centroids.begin();
         it != estimated_centroids.end(); it++) {
        geometry_msgs::Pose pose;
        pose.position.x = it->second(0);
        pose.position.y = it->second(1);
        pose.position.z = it->second(2);
        ecc_srv.request.estimated_centroids.push_back(pose);
    }
    ecc_srv.request.max_distance = static_cast<float>(this->eps_distance_);
    ecc_srv.request.min_samples = static_cast<int>(this->eps_min_samples_);
    if (this->clustering_client_.call(ecc_srv)) {
       int max_label = ecc_srv.response.argmax_label;
       if (max_label == -1) {
           return;
       }
       std::vector<uint32_t> bmi;
       for (int i = 0; i < ecc_srv.response.labels.size(); i++) {
          if (ecc_srv.response.indices[i] == max_label) {
             PointT pt;
             pt.x = ecc_srv.request.estimated_centroids[i].position.x;
             pt.y = ecc_srv.request.estimated_centroids[i].position.y;
             pt.z = ecc_srv.request.estimated_centroids[i].position.z;
             pt.g = 255;
             pt.b = 255;
             inliers->push_back(pt);
             bmi.push_back(best_match_index[i]);
          } else {
             PointT pt;
             pt.x = ecc_srv.request.estimated_centroids[i].position.x;
             pt.y = ecc_srv.request.estimated_centroids[i].position.y;
             pt.z = ecc_srv.request.estimated_centroids[i].position.z;
             pt.r = 255;
             pt.b = 255;
             inliers->push_back(pt);
          }
       }
       best_match_index.clear();
       best_match_index.insert(best_match_index.end(), bmi.begin(), bmi.end());
    } else {
       ROS_ERROR("ERROR! Failed to call Clustering Module\n");
       return;
    }
}

int main(int argc, char *argv[]) {

    ros::init(argc, argv, "multilayer_object_tracking");
    MultilayerObjectTracking mot;
    ros::spin();
    return 0;
}
