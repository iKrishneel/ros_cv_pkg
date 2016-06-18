
#include <cuboid_bilateral_symmetric_segmentation/object_region_handler.h>

ObjectRegionHandler::ObjectRegionHandler(const int mc_size, int thread) :
    min_cluster_size_(mc_size), num_threads_(thread), neigbor_size_(5) {
    this->in_cloud_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    this->sv_cloud_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    this->sv_normal_ = pcl::PointCloud<NormalT>::Ptr(
       new pcl::PointCloud<NormalT>);
    this->in_normals_ = pcl::PointCloud<NormalT>::Ptr(
       new pcl::PointCloud<NormalT>);
    this->region_indices_ = pcl::PointIndices::Ptr(new pcl::PointIndices);
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    this->all_indices_.clear();
    this->origin_ = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
}

bool ObjectRegionHandler::setInputCloud(
    const pcl::PointCloud<PointT>::Ptr cloud, std_msgs::Header header) {
    if (cloud->empty()) {
       ROS_ERROR("EMPTY CLOUD FOR REGION HANDLER");
       return false;
    }
    this->supervoxelSegmentation(cloud, this->supervoxel_clusters_,
                                 this->adjacency_list_);
    if (supervoxel_clusters_.empty()) {
       ROS_ERROR("SUPERVOXEL MAP IS EMPTY");
       return false;
    }
    
    if (this->angle_threshold_ > 0.0 && this->distance_threshold_ > 0.0) {
       this->supervoxelCoplanarityMerge(
          this->supervoxel_clusters_, this->adjacency_list_,
          this->angle_threshold_, this->distance_threshold_);
    }
    
    this->in_cloud_->clear();
    this->in_normals_->clear();
    this->indices_map_.clear();
    for (SupervoxelMap::iterator it = supervoxel_clusters_.begin();
         it != supervoxel_clusters_.end(); it++) {
       if (it->second->voxels_->size() > this->min_cluster_size_) {
          for (int i = 0; i < it->second->voxels_->size(); i++) {
             IndicesMap im;
             im.label = it->first;
             im.index = i;
             this->indices_map_.push_back(im);
          }
          *in_cloud_ += *(it->second->voxels_);
          *in_normals_ += *(it->second->normals_);
       }
    }

    if (this->in_cloud_->size() != this->in_normals_->size()) {
       ROS_ERROR("INCORRECT CLOUD AND NORMAL SIZE");
    }

    this->header_ = header;
    
    ROS_INFO("\033[34mPOINT CLOUD INFO IS SET FOR SEGMENTATION...\n\033[0m");
    return true;
}

/**
 * THE FUNCTION SELECTED SEED POINT ON SUPERVOXEL AND COMPUTES A
 * REGION AROUND THE SEED POINT. THE POINTS INDICES ARE SAVED FRO
 * PROCESSING ONCE THE OBJECT IS SEGMENTED
 */
bool ObjectRegionHandler::getCandidateRegion(
    SupervoxelMap &region_supervoxels,
    pcl::PointCloud<PointT>::Ptr out_cloud, pcl::PointXYZRGBNormal &info) {
    if (this->in_cloud_->size() < this->min_cluster_size_) {
       ROS_ERROR("INPUT CLOUD NOT SET");
       return false;
    }
    ROS_INFO("\033[34mSELECTING ONE CLUSTER\033[0m");
    double distance = DBL_MAX;
    uint32_t t_index = 0;

    pcl::PointCloud<PointT>::Ptr center_cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<NormalT>::Ptr center_normals(new pcl::PointCloud<NormalT>);
    int seed_index = -1;
    for (SupervoxelMap::iterator it = supervoxel_clusters_.begin();
         it != supervoxel_clusters_.end(); it++) {
       Eigen::Vector4f centroid = it->second->centroid_.getVector4fMap();
       centroid(3) = 1.0f;
       origin_(3) = 1.0f;
       double dist = pcl::distances::l2(origin_, centroid);

       if (dist < distance && it->second->voxels_->size() >
           this->min_cluster_size_ && dist > 0.05f) {
          distance = dist;
          t_index = it->first;
          seed_index = static_cast<int>(center_cloud->size());
       }
       PointT seed_point;
       seed_point.x = it->second->centroid_.x;
       seed_point.y = it->second->centroid_.y;
       seed_point.z = it->second->centroid_.z;
       seed_point.r = it->second->centroid_.r;
       seed_point.g = it->second->centroid_.g;
       seed_point.b = it->second->centroid_.b;
       center_cloud->push_back(seed_point);
       center_normals->push_back(it->second->normal_);
    }
    if (seed_index == -1) {
       ROS_ERROR("CANNOT SELECT ANY SUPERVOXEL");
       return false;
    }
    origin_ = supervoxel_clusters_.at(t_index)->centroid_.getVector4fMap();
    
    seed_point_.x = supervoxel_clusters_.at(t_index)->centroid_.x;
    seed_point_.y = supervoxel_clusters_.at(t_index)->centroid_.y;
    seed_point_.z = supervoxel_clusters_.at(t_index)->centroid_.z;
    seed_point_.r = supervoxel_clusters_.at(t_index)->centroid_.r;
    seed_point_.g = supervoxel_clusters_.at(t_index)->centroid_.g;
    seed_point_.b = supervoxel_clusters_.at(t_index)->centroid_.b;
    seed_normal_ = supervoxel_clusters_.at(t_index)->normal_;
    
    ROS_INFO("\033[34mDOING REGION GROWING\033[0m");

    std::vector<int> labels(static_cast<int>(center_cloud->size()));
    for (int i = 0; i < center_cloud->size(); i++) {
       if (i == seed_index) {
          labels[i] = 1;
       } else {
          labels[i] = -1;
       }
    }
    this->kdtree_->setInputCloud(center_cloud);
    this->seedCorrespondingRegion(labels, center_cloud,
                                  center_normals, seed_index);
    
    pcl::PointCloud<PointT>::Ptr seed_cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<NormalT>::Ptr seed_normals(new pcl::PointCloud<NormalT>);
    this->region_indices_->indices.clear();

    int icounter = 0;
    this->convex_supervoxel_clusters_.clear();
    double cluster_distances = 0.0;
    Eigen::Vector4f seed_pt = this->seed_point_.getVector4fMap();
    seed_pt(3) = 1.0f;
    for (SupervoxelMap::iterator it = supervoxel_clusters_.begin();
         it != supervoxel_clusters_.end(); it++) {

       std::cout << labels[icounter]  << ", ";
       
       if (labels[icounter++] != -1) {
          *seed_cloud += *(it->second->voxels_);
          *seed_normals += *(it->second->normals_);
          Eigen::Vector4f center = it->second->centroid_.getVector4fMap();
          center(3) = 1.0f;
          double d = pcl::distances::l2(seed_pt, center);
          if (d > cluster_distances) {
             cluster_distances = d;
          }
          pcl::Supervoxel<PointT>::Ptr super_v(new pcl::Supervoxel<PointT>);
          for (int i = 0; i < it->second->voxels_->size(); i++) {
             super_v->voxels_->push_back(it->second->voxels_->points[i]);
             super_v->normals_->push_back(it->second->normals_->points[i]);
          }
          this->convex_supervoxel_clusters_[it->first] = super_v;
       }
    }
    
    //! oversegment supervoxels
    /*
    seed_cloud->clear();
    seed_normals->clear();
    this->region_indices_->indices.clear();
    for (SupervoxelMap::iterator it = supervoxel_clusters_.begin();
         it != supervoxel_clusters_.end(); it++) {
       Eigen::Vector4f center = it->second->centroid_.getVector4fMap();
       center(3) = 1.0f;
       double dist = pcl::distances::l2(seed_pt, center);
       if (dist < cluster_distances) {

          int csize = static_cast<int>(seed_cloud->size());
          for (int i = 0; i < it->second->voxels_->size(); i++) {
             this->region_indices_->indices.push_back(i + csize);
          }
          
          *seed_cloud += *(it->second->voxels_);
          *seed_normals += *(it->second->normals_);
       }
    }
    */

    
    std::cout << "\nSIZE: " << seed_cloud->size()  << "\t"
              << region_indices_->indices.size()  << "\n";
    
    this->kdtree_->setInputCloud(this->in_cloud_);
    this->regionOverSegmentation(seed_cloud, seed_normals,
                                 this->in_cloud_, this->in_normals_);

    ROS_INFO("\033[34mEXTRACTING SUPERVOXEL: %d\033[0m", seed_cloud->size());
    
    this->getRegionSupervoxels(region_supervoxels, seed_cloud);
    
    out_cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*seed_cloud, *out_cloud);

    //! set seed points to initial one ? find better way
    seed_point_ = center_cloud->points[seed_index];
    seed_normal_ = center_normals->points[seed_index];
    
    info.x = this->seed_point_.x;
    info.y = this->seed_point_.y;
    info.z = this->seed_point_.z;
    info.r = this->seed_point_.r;
    info.b = this->seed_point_.b;
    info.g = this->seed_point_.g;
    info.normal_x = this->seed_normal_.normal_x;
    info.normal_y = this->seed_normal_.normal_y;
    info.normal_z = this->seed_normal_.normal_z;
    info.curvature = this->seed_normal_.curvature;

    ROS_INFO("\033[34mDONE: %d\033[0m", out_cloud->size());

    return true;
}

void ObjectRegionHandler::updateObjectRegion(
    pcl::PointCloud<PointT>::Ptr cloud, const pcl::PointIndices::Ptr labels) {
    if (cloud->empty() || labels->indices.empty()) {
       return;
    }
    ROS_INFO("\033[36mDOING CLUSTERING\033[0m");
    
    std::vector<pcl::PointIndices> cluster_indices;
    this->doEuclideanClustering(cluster_indices, cloud, labels, 0.005f,
                                this->min_cluster_size_);
    if (cluster_indices.empty()) {
       return;
    }
    int object_index = -1;
    double distance = DBL_MAX;
    pcl::PointCloud<PointT>::Ptr temp_cloud(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(*cloud, *temp_cloud);
    cloud->clear();
    for (int i = 0; i < cluster_indices.size(); i++) {

       std::cout << "\033[33m SIZE: \033[0m" << cluster_indices[
          i].indices.size()  << "\n";
       
       pcl::PointCloud<PointT>::Ptr temp(new pcl::PointCloud<PointT>);
       for (int j = 0; j < cluster_indices[i].indices.size(); j++) {
          int idx = cluster_indices[i].indices[j];
          temp->push_back(temp_cloud->points[idx]);  //! bug
       }
       
       Eigen::Vector4f centroid;
       pcl::compute3DCentroid<PointT, float>(*temp, centroid);
       centroid(3) = 1.0f;
       Eigen::Vector4f seed_point = seed_point_.getVector4fMap();
       seed_point(3) = 1.0f;
       double d = pcl::distances::l2(centroid, seed_point);
       if (d < distance) {
          distance = d;
          cloud->clear();
          pcl::copyPointCloud<PointT, PointT>(*temp, *cloud);
          object_index = i;
       }
    }
    
    if (object_index == -1) {
       ROS_ERROR("NO REGION FOUND");
       return;
    }
    
    ROS_INFO("\033[36mREMOVING POINTS ON SUPERVOXEL MAP\033[0m");

    //! target object cloud
    pcl::PointIndices seed_region_indices;
    int s_size = static_cast<int>(sv_cloud_->size());
    for (int i = 0; i < cluster_indices[object_index].indices.size(); i++) {
       int indx = cluster_indices[object_index].indices[i];
       indx = this->region_indices_->indices[indx];
       //! get label of sv
       uint32_t label = indices_map_[indx].label;
       int pt_index = indices_map_[indx].index;
       
       //! prune points and normals of sv
       PointT pt = this->in_cloud_->points[indx];
       pt.x = std::numeric_limits<float>::quiet_NaN();
       pt.y = std::numeric_limits<float>::quiet_NaN();
       pt.z = std::numeric_limits<float>::quiet_NaN();
       this->supervoxel_clusters_.at(label)->voxels_->points[pt_index] = pt;
       
       seed_region_indices.indices.push_back(i + s_size);
       this->sv_cloud_->push_back(this->in_cloud_->points[indx]);
       this->sv_normal_->push_back(this->in_normals_->points[indx]);

       //! new sv input
       this->in_cloud_->points[indx] = pt;
    }
    this->all_indices_.push_back(seed_region_indices);
    
    ROS_INFO("\033[36mUPDATING CLOUD\033[0m");
    
    this->setInputCloud(in_cloud_, this->header_);
    return;


    
    // TODO(BUG): FIX THE BUG TO REMOVE SMALL SUPERVOXELS
    //! update the normals and centroids of the voxels
    this->in_cloud_->clear();
    this->in_normals_->clear();
    this->indices_map_.clear();
    for (SupervoxelMap::iterator it = this->supervoxel_clusters_.begin();
         it != this->supervoxel_clusters_.end(); it++) {
       pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
       std::vector<int> nan_indices;
       for (int i = 0; i < it->second->voxels_->size(); i++) {
          PointT pt = it->second->voxels_->points[i];
          if (!isnan(pt.x) || !isnan(pt.y) || !isnan(pt.z)) {
             nan_indices.push_back(i);
             tmp_cloud->push_back(pt);
          }
       }
       it->second->voxels_->clear();
       
       if (tmp_cloud->size() < this->min_cluster_size_) {
          // TODO(HERE):  move this points to object region
          
       } else {
          pcl::copyPointCloud<PointT, PointT>(
             *tmp_cloud, *(it->second->voxels_));
          
          pcl::PointCloud<NormalT>::Ptr tmp_norm(new pcl::PointCloud<NormalT>);
          PointT n_center;
          int y = 0;
          for (int i = 0; i < nan_indices.size(); i++) {
             int indx = nan_indices[i];
             tmp_norm->push_back(it->second->normals_->points[indx]);

             n_center.x += tmp_cloud->points[y].x;
             n_center.y += tmp_cloud->points[y].y;
             n_center.z += tmp_cloud->points[y].z;
             y++;

             IndicesMap im;
             im.label = it->first;
             im.index = i;
             this->indices_map_.push_back(im);
          }
          it->second->centroid_.x = n_center.x / static_cast<float>(y);
          it->second->centroid_.y = n_center.y / static_cast<float>(y);
          it->second->centroid_.z = n_center.z / static_cast<float>(y);
          
          pcl::copyPointCloud<NormalT, NormalT>(
             *tmp_norm, *(it->second->normals_));

          *in_cloud_ += *it->second->voxels_;
          *in_normals_ += *it->second->normals_;
       }
    }

    std::cout << "\n\n INFO: "  << "\n";
    std::cout << in_cloud_->size() << "\t" << in_normals_->size()  << "\n";

    this->kdtree_->setInputCloud(this->in_cloud_);
    cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*in_cloud_, *cloud);
}

void ObjectRegionHandler::getLabels(
    std::vector<pcl::PointIndices> &all_indices) {
    all_indices.clear();
    all_indices = this->all_indices_;
}

void ObjectRegionHandler::seedCorrespondingRegion(
    std::vector<int> &labels, const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<NormalT>::Ptr normals, const int parent_index) {
    std::vector<int> neigbor_indices;
    this->pointNeigbour<int>(neigbor_indices, cloud->points[parent_index],
                             this->neigbor_size_);

    int neigb_lenght = static_cast<int>(neigbor_indices.size());
    std::vector<int> merge_list(neigb_lenght);
    merge_list[0] = -1;
    
    for (int i = 1; i < neigbor_indices.size(); i++) {
        int index = neigbor_indices[i];
        if (index != parent_index && labels[index] == -1) {
           Eigen::Vector4f parent_pt = this->seed_point_.getVector4fMap();
           parent_pt(3) = 1.0f;
           Eigen::Vector4f parent_norm =
              this->seed_normal_.getNormalVector4fMap();
           Eigen::Vector4f child_pt = cloud->points[index].getVector4fMap();
           child_pt(3) = 1.0f;
           Eigen::Vector4f child_norm = normals->points[
              index].getNormalVector4fMap();
           if (this->seedVoxelConvexityCriteria(
                  parent_pt, parent_norm, child_pt, child_norm, -0.01f) == 1) {
              merge_list[i] = index;
              labels[index] = 1;
           } else {
              merge_list[i] = -1;
           }
        } else {
           merge_list[i] = -1;
        }
    }

    for (int i = 0; i < merge_list.size(); i++) {
       int index = merge_list[i];
       if (index != -1) {
          seedCorrespondingRegion(labels, cloud, normals, index);
       }
    }
}

int ObjectRegionHandler::seedVoxelConvexityCriteria(
    Eigen::Vector4f c_centroid, Eigen::Vector4f c_normal,
    Eigen::Vector4f n_centroid, Eigen::Vector4f n_normal,
    const float thresh, bool is_seed) {
    float pt2seed_relation = FLT_MAX;
    float seed2pt_relation = FLT_MAX;
    if (is_seed) {
       pt2seed_relation = (n_centroid - c_centroid).dot(n_normal);
       seed2pt_relation = (c_centroid - n_centroid).dot(c_normal);
    }
    float norm_similarity = (M_PI - std::acos(
                                c_normal.dot(n_normal) /
                                (c_normal.norm() * n_normal.norm()))) / M_PI;
    if (seed2pt_relation > thresh && pt2seed_relation > thresh) {
       return 1;
    } else {
       return -1;
    }
}

template<class T>
void ObjectRegionHandler::estimateNormals(
    const pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<NormalT>::Ptr normals, const T k, bool use_knn) const {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: The Input cloud is Empty.....");
       return;
    }
    // TODO(HERE):  MOVE TO GLOBAL
    pcl::NormalEstimationOMP<PointT, NormalT> ne;
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

template<class T>
void ObjectRegionHandler::pointNeigbour(
    std::vector<int> &neigbor_indices, const PointT seed_point,
    const T K, bool is_knn) {
    if (isnan(seed_point.x) || isnan(seed_point.y) || isnan(seed_point.z)) {
       ROS_ERROR("SEED POINT FOR KNN IS NAN");
       return;
    }
    neigbor_indices.clear();
    std::vector<float> point_squared_distance;
    if (is_knn) {
       int search_out = this->kdtree_->nearestKSearch(
          seed_point, K, neigbor_indices, point_squared_distance);
    } else {
       int search_out = this->kdtree_->radiusSearch(
          seed_point, K, neigbor_indices, point_squared_distance);
    }
}

void ObjectRegionHandler::regionOverSegmentation(
    pcl::PointCloud<PointT>::Ptr region, pcl::PointCloud<NormalT>::Ptr normal,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<NormalT>::Ptr normals) {
    if (cloud->empty() || region->empty() || normals->empty()) {
       ROS_ERROR("EMPTY CLOUD FOR REGION OVERSEGMENTATION");
       return;
    }
    Eigen::Vector4f center;
    pcl::compute3DCentroid<PointT, float>(*region, center);
    double distance = 0.15;  // compute max distace from center
    center(3) = 1.0f;
    for (int i = 0; i < region->size(); i++) {
       double d = pcl::distances::l2(region->points[i].getVector4fMap(),
                                     center);
       if (d > distance) {
          distance = d;
       }
    }
    region->clear();
    normal->clear();
    this->region_indices_->indices.clear();
    
    double dist = DBL_MAX;
    int idx = -1;
    int icount = 0;
    pcl::PointIndices::Ptr prob_indices(new pcl::PointIndices);
    Eigen::Vector4f seed_point = this->seed_point_.getVector4fMap();
    seed_point(3) = 1.0f;
    for (int i = 0; i < cloud->size(); i++) {
       if (pcl::distances::l2(cloud->points[i].getVector4fMap(),
                              center) < distance) {
          PointT pt = cloud->points[i];
          if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z)) {
             region->push_back(cloud->points[i]);
             normal->push_back(normals->points[i]);
             prob_indices->indices.push_back(i);

             this->region_indices_->indices.push_back(i);
             
             // update the seed info
             Eigen::Vector4f cloud_pt = cloud->points[i].getVector4fMap();
             cloud_pt(3) = 1.0f;
             double d = pcl::distances::l2(cloud_pt, seed_point);
             if (d < dist) {
                dist = d;
                idx = icount;
             }
             icount++;
          }
       }
    }

    ROS_ERROR("\n INDICES SIZE: %d", prob_indices->indices.size());
    
    bool is_cluster = !true;
    if (is_cluster) {

       ROS_INFO("\033[33m \t\tDOIING CLUSTERING \033[0m");
       
       
       std::vector<pcl::PointIndices> cluster_indices;
       cluster_indices.clear();
       pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
       tree->setInputCloud(cloud);
       pcl::EuclideanClusterExtraction<PointT> euclidean_clustering;
       euclidean_clustering.setClusterTolerance(0.01f);
       euclidean_clustering.setMinClusterSize(this->min_cluster_size_);
       euclidean_clustering.setMaxClusterSize(250000);
       euclidean_clustering.setSearchMethod(tree);
       euclidean_clustering.setInputCloud(cloud);
       euclidean_clustering.setIndices(prob_indices);
       euclidean_clustering.extract(cluster_indices);
       double min_distance = DBL_MAX;
       idx = -1;
       dist = DBL_MAX;

       ROS_INFO("\033[33m \t\tSELECTING BEST \033[0m");
       
       for (int i = 0; i < cluster_indices.size(); i++) {
          pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
          pcl::PointCloud<NormalT>::Ptr tmp_normal(
             new pcl::PointCloud<NormalT>);
          for (int j = 0; j < cluster_indices[i].indices.size(); j++) {
             int cidx = cluster_indices[i].indices[j];
             tmp_cloud->push_back(cloud->points[cidx]);
             tmp_normal->push_back(normals->points[cidx]);
          }
          Eigen::Vector4f center;
          pcl::compute3DCentroid<PointT, float>(*tmp_cloud, center);
          center(3) = 1.0f;
          double d = pcl::distances::l2(seed_point, center);
          if (d < dist) {
             dist = d;
             idx = i;

             region->clear();
             pcl::copyPointCloud<PointT, PointT>(*tmp_cloud, *region);
             normal->clear();
             pcl::copyPointCloud<NormalT, NormalT>(*tmp_normal, *normal);
          }
       }

       ROS_INFO("\033[33m \t\tCOPYING ....%d \033[0m", idx);
       
       // this->region_indices_->indices.clear();
       // for (int i = 0; i < cluster_indices[idx].indices.size(); i++) {
       //    int pt_indx = cluster_indices[idx].indices[i];
       //    this->region_indices_->indices.push_back(pt_indx);
       // }

       //! update seed point
       // ROS_WARN("ENABLE: SEED POINT INFO UPDATED");
       
       // int csize = cluster_indices[idx].indices.size() / 2;
       // int ind = cluster_indices[idx].indices[csize];
       // this->seed_point_ = cloud->points[ind];
       // this->seed_normal_ = normals->points[ind];
    }
    
    // TODO(HERE):  if lenght is small merge fix size
}

void ObjectRegionHandler::doEuclideanClustering(
    std::vector<pcl::PointIndices> &cluster_indices,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointIndices::Ptr prob_indices, const float tolerance_thresh,
    const int min_size_thresh, const int max_size_thresh) {
    if (cloud->empty()) {
       ROS_ERROR("EMPTY CLOUD FOR CLUSTERING");
       return;
    }
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);
    pcl::EuclideanClusterExtraction<PointT> euclidean_clustering;
    euclidean_clustering.setClusterTolerance(tolerance_thresh);
    euclidean_clustering.setMinClusterSize(min_size_thresh);
    euclidean_clustering.setMaxClusterSize(max_size_thresh);
    euclidean_clustering.setSearchMethod(tree);
    euclidean_clustering.setInputCloud(cloud);
    if (!prob_indices->indices.empty()) {
       euclidean_clustering.setIndices(prob_indices);
    }
    cluster_indices.clear();
    euclidean_clustering.extract(cluster_indices);
}

void ObjectRegionHandler::getRegionSupervoxels(
    SupervoxelMap &region_supervoxels, pcl::PointCloud<PointT>::Ptr region) {
    if (supervoxel_clusters_.empty() || region_indices_->indices.empty()) {
       ROS_ERROR("SUPERVOXEL OR INDEX ARE EMPTY. \nNO REGION CREATED ...");
       std::cout << supervoxel_clusters_.size() << "\t"
                 << region_indices_->indices.size() << "\n";
       return;
    }
    UInt32Map region_cache;
    for (SupervoxelMap::const_iterator it = supervoxel_clusters_.begin();
         it != supervoxel_clusters_.end(); it++) {
       region_cache[it->first] = 0;
    }
    
    for (int i = 0; i < this->region_indices_->indices.size(); i++) {
       int index = this->region_indices_->indices[i];
       uint32_t label = this->indices_map_[index].label;
       int pt_indx = this->indices_map_[index].index;
       region_cache[label]++;
    }

    region_supervoxels.clear();
    region->clear();
    SupervoxelMap copy_region_sv;
    this->region_indices_->indices.clear();
    pcl::PointCloud<NormalT>::Ptr region_normal(new pcl::PointCloud<NormalT>);
    for (UInt32Map::iterator it = region_cache.begin();
         it != region_cache.end(); it++) {
       int v_size = supervoxel_clusters_.at(it->first)->voxels_->size() / 2;
       if (it->second > v_size) {
          pcl::Supervoxel<PointT>::Ptr sv = this->supervoxel_clusters_.at(
             it->first);
          pcl::Supervoxel<PointT>::Ptr super_v(new pcl::Supervoxel<PointT>);
          for (int i = 0; i < sv->voxels_->size(); i++) {
             PointT pt = sv->voxels_->points[i];
             if (isnan(pt.x) || isnan(pt.y) || isnan(pt.z)) {
                std::cout << "\033[31m Point is nan:  \033[0m"  << i  << "\n";
             }
             super_v->voxels_->push_back(sv->voxels_->points[i]);
             super_v->normals_->push_back(sv->normals_->points[i]);
          }
          super_v->centroid_ = sv->centroid_;
          super_v->normal_ = sv->normal_;
          region_supervoxels[it->first] = super_v;
          copy_region_sv[it->first] = super_v;
          
          *region += *(supervoxel_clusters_.at(it->first)->voxels_);
          *region_normal += *(supervoxel_clusters_.at(it->first)->normals_);
          
          for (int j = 0; j < this->indices_map_.size(); j++) {
             IndicesMap im = this->indices_map_[j];
             if (im.label == it->first) {
                this->region_indices_->indices.push_back(j);
             }
          }
       } else if (it->second < v_size) {
          continue;
       }
    }

    //! update seed
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid<PointT, float>(*region, centroid);
    PointT seed_point;
    seed_point.x = centroid(0);
    seed_point.y = centroid(1);
    seed_point.z = centroid(2);
    std::vector<int> neigbor_indices;
    this->pointNeigbour<int>(neigbor_indices, seed_point, 1);
    if (neigbor_indices.empty()) {
       int csize = region->size()/2;
       this->seed_point_ = region->points[csize];
       this->seed_normal_ = region_normal->points[csize];
    }
    int idx = neigbor_indices[0];
    // Eigen::Vector4f cpt = seed_point.getVector4fMap();
    // Eigen::Vector4f npt = in_cloud_->points[idx].getVector4fMap();
    // cpt(3) = 1.0f;
    // npt(3) = 1.0f;
    // double d = pcl::distances::l2(cpt, npt);
    // if (d < this->seed_resolution_ * 4.0) {
    this->seed_index_ = idx;
    this->seed_point_ = this->in_cloud_->points[idx];
    this->seed_normal_ = this->in_normals_->points[idx];
    // } else {
    //    int csize = region->size()/2;
    //    this->seed_point_ = region->points[csize];
    //    this->seed_normal_ = region_normal->points[csize];
    // }
    
       
}


void ObjectRegionHandler::supervoxelCoplanarityMerge(
    SupervoxelMap &supervoxel_clusters, AdjacencyList &adjacency_list,
    const float angle_threshold, const float distance_threshold) {
    if (supervoxel_clusters.empty()) {
       ROS_ERROR("EMPTY SUPERVOXEL FOR MERGING");
       return;
    }
    SupervoxelMap coplanar_supervoxels;
    UInt32Map voxel_labels;
    coplanar_supervoxels.clear();
    for (SupervoxelMap::iterator it = supervoxel_clusters.begin();
         it != supervoxel_clusters.end(); it++) {
       voxel_labels[it->first] = -1;
    }
    int label = -1;
    AdjacencyList::vertex_iterator i, end;
    for (boost::tie(i, end) = boost::vertices(adjacency_list); i != end; i++) {
       AdjacencyList::adjacency_iterator ai, a_end;
       boost::tie(ai, a_end) = boost::adjacent_vertices(*i, adjacency_list);
       uint32_t vindex = static_cast<int>(adjacency_list[*i]);
       UInt32Map::iterator it = voxel_labels.find(vindex);
       if (it->second == -1) {
          voxel_labels[vindex] = ++label;
       }
       bool vertex_has_neigbor = true;
       if (ai == a_end) {
          vertex_has_neigbor = false;
          if (!supervoxel_clusters.at(vindex)->voxels_->empty()) {
             coplanar_supervoxels[vindex] = supervoxel_clusters.at(vindex);
          }
       }
       Eigen::Vector4f v_normal = supervoxel_clusters.at(
          vindex)->normal_.getNormalVector4fMap();
       Eigen::Vector4f v_centroid = supervoxel_clusters.at(
          vindex)->centroid_.getVector4fMap();
       
       while (vertex_has_neigbor) {
          bool found = false;
          AdjacencyList::edge_descriptor e_descriptor;
          boost::tie(e_descriptor, found) = boost::edge(
             *i, *ai, adjacency_list);
          if (found) {
             float weight = adjacency_list[e_descriptor];
             uint32_t n_vindex = adjacency_list[*ai];

             Eigen::Vector4f n_normal = supervoxel_clusters.at(
               n_vindex)->normal_.getNormalVector4fMap();
             Eigen::Vector4f n_centroid = supervoxel_clusters.at(
                n_vindex)->centroid_.getVector4fMap();
             float coplanar_criteria = this->coplanarityCriteria(
                v_centroid, n_centroid, v_normal, n_normal,
                angle_threshold, distance_threshold);
             
             if (coplanar_criteria <= static_cast<float>(coplanar_threshold_)) {
                boost::remove_edge(e_descriptor, adjacency_list);
             } else {
                this->updateSupervoxelClusters(supervoxel_clusters,
                                               vindex, n_vindex);
                
                AdjacencyList::adjacency_iterator ni, n_end;
                boost::tie(ni, n_end) = boost::adjacent_vertices(
                   *ai, adjacency_list);
                
                for (; ni != n_end; ni++) {
                   bool is_found = false;
                   AdjacencyList::edge_descriptor n_edge;
                   boost::tie(n_edge, is_found) = boost::edge(
                      *ai, *ni, adjacency_list);
                   if (is_found && (*ni != *i)) {
                      boost::add_edge(*i, *ni, FLT_MIN, adjacency_list);
                   } else if (is_found && (*ni == *i)) {
                      continue;
                   }
                }
                boost::clear_vertex(*ai, adjacency_list);
                voxel_labels[n_vindex] = label;
             }
          }
          
          boost::tie(ai, a_end) = boost::adjacent_vertices(*i, adjacency_list);
          if (ai == a_end) {
             coplanar_supervoxels[vindex] = supervoxel_clusters.at(vindex);
             vertex_has_neigbor = false;
          } else {
             vertex_has_neigbor = true;
          }
       }
    }
    supervoxel_clusters.clear();
    supervoxel_clusters = coplanar_supervoxels;
    // SupervoxelMap().swap(coplanar_supervoxels);
}

void ObjectRegionHandler::updateSupervoxelClusters(
    SupervoxelMap &supervoxel_clusters, const uint32_t vindex,
    const uint32_t n_vindex) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    *cloud = *(supervoxel_clusters.at(vindex)->voxels_) +
       *(supervoxel_clusters.at(n_vindex)->voxels_);
    pcl::PointCloud<pcl::Normal>::Ptr normals(
       new pcl::PointCloud<pcl::Normal>);
    *normals = *(supervoxel_clusters.at(vindex)->normals_) +
       *(supervoxel_clusters.at(n_vindex)->normals_);
    Eigen::Vector4f centre;
    pcl::compute3DCentroid<PointT, float>(*cloud, centre);
    pcl::PointXYZRGBA centroid;
    centroid.x = centre(0);
    centroid.y = centre(1);
    centroid.z = centre(2);
    pcl::PointXYZRGBA vcent = supervoxel_clusters.at(vindex)->centroid_;
    pcl::PointXYZRGBA n_vcent = supervoxel_clusters.at(n_vindex)->centroid_;
    centroid.g = (vcent.g - n_vcent.g)/2 + n_vcent.g;
    centroid.b = (vcent.b - n_vcent.b)/2 + n_vcent.b;
    centroid.a = (vcent.a - n_vcent.a)/2 + n_vcent.a;
    supervoxel_clusters.at(vindex)->voxels_->clear();
    supervoxel_clusters.at(vindex)->normals_->clear();
    *(supervoxel_clusters.at(vindex)->voxels_) = *cloud;
    *(supervoxel_clusters.at(vindex)->normals_) = *normals;
    supervoxel_clusters.at(vindex)->centroid_ = centroid;

    supervoxel_clusters.at(n_vindex)->voxels_->clear();
    supervoxel_clusters.at(n_vindex)->normals_->clear();
}

float ObjectRegionHandler::coplanarityCriteria(
    const Eigen::Vector4f centroid, const Eigen::Vector4f n_centroid,
    const Eigen::Vector4f normal, const Eigen::Vector4f n_normal,
    const float angle_thresh, const float dist_thresh) {
    float tetha = std::acos(normal.dot(n_normal) / (
                              n_normal.norm() * normal.norm()));
    float ang_thresh = angle_thresh * (M_PI/180.0f);
    float coplannarity = 0.0f;
    if (tetha < ang_thresh) {
       float direct1 = normal.dot(centroid - n_centroid);
       float direct2 = n_normal.dot(centroid - n_centroid);
       float dist = std::fabs(std::max(direct1, direct2));
       if (dist < dist_thresh) {
          coplannarity = 1.0f;
       }
    }
    return coplannarity;
}

