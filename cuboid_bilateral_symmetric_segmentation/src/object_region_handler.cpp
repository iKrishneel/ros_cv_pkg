
#include <cuboid_bilateral_symmetric_segmentation/object_region_handler.h>

ObjectRegionHandler::ObjectRegionHandler(const int mc_size, int thread) :
    min_cluster_size_(mc_size), is_init_(true), num_threads_(thread),
    neigbor_size_(50) {
    this->in_cloud_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    this->in_normals_ = pcl::PointCloud<NormalT>::Ptr(
       new pcl::PointCloud<NormalT>);
    this->region_indices_ = pcl::PointIndices::Ptr(new pcl::PointIndices);
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
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
    this->in_cloud_->clear();
    this->in_normals_->clear();
    this->indices_map_.clear();
    for (SupervoxelMap::iterator it = supervoxel_clusters_.begin();
         it != supervoxel_clusters_.end(); it++) {
       int s_size = static_cast<int>(this->in_cloud_->size());
       for (int i = 0; i < it->second->voxels_->size(); i++) {
          IndicesMap im;
          im.label = it->first;
          im.index = i;
          this->indices_map_.push_back(im);
       }
       *in_cloud_ += *(it->second->voxels_);
       *in_normals_ += *(it->second->normals_);
    }

    if (this->in_cloud_->size() != this->in_normals_->size()) {
       ROS_ERROR("INCORRECT CLOUD AND NORMAL SIZE");
    }
    
    this->kdtree_->setInputCloud(this->in_cloud_);
    this->is_init_ = true;
    this->header_ = header;
    this->all_indices_.clear();
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
    //! select one supervoxel
    uint32_t t_index = supervoxel_clusters_.size() + 1;
    for (SupervoxelMap::iterator it = supervoxel_clusters_.begin();
         it != supervoxel_clusters_.end(); it++) {
       if (it->second->voxels_->size() > this->min_cluster_size_) {
          t_index = it->first;
          break;
       }
    }
    
    if (t_index == supervoxel_clusters_.size() + 1) {
       ROS_ERROR("CANNOT SELECT ANY SUPERVOXEL");
       return false;
    } else {
       PointT seed_point;
       seed_point.x = supervoxel_clusters_.at(t_index)->centroid_.x;
       seed_point.y = supervoxel_clusters_.at(t_index)->centroid_.y;
       seed_point.z = supervoxel_clusters_.at(t_index)->centroid_.z;
       seed_point.r = supervoxel_clusters_.at(t_index)->centroid_.r;
       seed_point.g = supervoxel_clusters_.at(t_index)->centroid_.g;
       seed_point.b = supervoxel_clusters_.at(t_index)->centroid_.b;
       int k = 1;
       std::vector<int> neigbor_indices;

       std::cout << seed_point  << "\n";
       
       this->pointNeigbour<int>(neigbor_indices, seed_point, k);
       if (neigbor_indices.empty()) {
          ROS_ERROR("NO NEAREST NEIGBOIR TO SEED POINT FOUND");
          return false;
       }
       int idx = neigbor_indices[0];
       Eigen::Vector4f cpt = seed_point.getVector4fMap();
       Eigen::Vector4f npt = in_cloud_->points[idx].getVector4fMap();
       cpt(3) = 1.0f;
       npt(3) = 1.0f;
       double d = pcl::distances::l2(cpt, npt);
       if (d < this->seed_resolution_ * 4.0) {
          this->seed_index_ = idx;
          this->seed_point_ = this->in_cloud_->points[idx];
          this->seed_normal_ = this->in_normals_->points[idx];
       } else {
          ROS_ERROR("NEIGBOR TOO FAR: %3.2f, %f", d, this->seed_resolution_);
          return false;
       }
    }
    ROS_INFO("\033[34mDOING REGION GROWING\033[0m");
    
    //! region growing
    std::vector<int> labels(static_cast<int>(this->in_cloud_->size()));
    for (int i = 0; i < this->in_cloud_->size(); i++) {
       if (i == this->seed_index_) {
          labels[i] = 1;
       }
       labels[i] = -1;
    }
    this->seedCorrespondingRegion(labels, this->in_cloud_,
                                  this->in_normals_, this->seed_index_);

    pcl::PointCloud<PointT>::Ptr seed_cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<NormalT>::Ptr seed_normals(new pcl::PointCloud<NormalT>);
    this->region_indices_->indices.clear();
    for (int i = 0; i < labels.size(); i++) {
       if (labels[i] != -1) {
          PointT pt = this->in_cloud_->points[i];
          seed_cloud->push_back(pt);
          seed_normals->push_back(in_normals_->points[i]);

          // this->region_indices_->indices.push_back(i);
       }
    }

    std::cout << "\nSEED_SIZE: " << seed_cloud->size() << "\t"
              << seed_normals->size()  << "\n";
    
    //! get lenght
    this->regionOverSegmentation(seed_cloud, seed_normals,
                                 this->in_cloud_, this->in_normals_);

    // get supervoxel
    this->getRegionSupervoxels(region_supervoxels, seed_cloud);

    /*
    for (SupervoxelMap::iterator it = region_supervoxels.begin();
         it != region_supervoxels.end(); it++) {
       pcl::Supervoxel<PointT>::Ptr super_v(new pcl::Supervoxel<PointT>);
       for (int i = 0; i < it->second->voxels_->size(); i++) {
          super_v->voxels_->push_back(it->second->voxels_->points[i]);
          super_v->normals_->push_back(it->second->normals_->points[i]);
       }
       super_v->centroid_ = it->second->centroid_;
       super_v->normal_ = it->second->normal_;
       
       // this->region_supervoxels_[it->first] = it->second;
       this->region_supervoxels_[it->first] = super_v;
       
       std::cout << "\t\t" << it->second->voxels_->size()  << "\t"
                 << region_supervoxels.size() <<"\n";
    }
    */
    
    
    out_cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*seed_cloud, *out_cloud);
    
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
    // std::cout << "INDICES SIZE: " << region_indices_->indices.size()  << "\n";

    if (iter_counter_-- == 0) {
       return false;
    }
    return true;
}


/**
 * FUNCTION RECEVICES THE POINT INDICES OF OBJECT (1) AND NON-OBJECT
 * (-1) AND REMOVED THE POINTS FROM THE SUPERVOXEL AND THE POINT CLOUD
 */
void ObjectRegionHandler::updateObjectRegion(
    pcl::PointCloud<PointT>::Ptr cloud, const pcl::PointIndices::Ptr labels) {
    if (cloud->empty() /*|| labels->indices.empty()*/) {
       return;
    }

    /**
     * region_indices_ and labels should be of same size
     */

    ROS_INFO("\033[36mDOING CLUSTERING\033[0m");

    std::cout << "input cloud: " << cloud->size()  << "\n";
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::PointIndices::Ptr prob_indices(new pcl::PointIndices);
    this->doEuclideanClustering(cluster_indices, cloud, prob_indices);

    if (cluster_indices.empty()) {
       return;
    }
    
    ROS_INFO("\033[36mGETTING THE BEST CLOUD\033[0m");
    std::cout << "cluster size: " << cluster_indices.size()  << "\n";

    int object_index = -1;
    double distance = DBL_MAX;
    for (int i = 0; i < cluster_indices.size(); i++) {
       pcl::PointCloud<PointT>::Ptr temp(new pcl::PointCloud<PointT>);
       for (int j = 0; j < cluster_indices[i].indices.size(); j++) {
          int idx = cluster_indices[i].indices[j];
          temp->push_back(cloud->points[idx]);
       }
       
       Eigen::Vector4f centroid;
       pcl::compute3DCentroid<PointT, float>(*temp, centroid);
       double d = pcl::distances::l2(centroid, seed_point_.getVector4fMap());
       if (d < distance) {
          distance = d;
          cloud->clear();
          pcl::copyPointCloud<PointT, PointT>(*temp, *cloud);
          object_index = i;
       }
    }
    
    ROS_INFO("\033[36mREMOVING POINTS ON SUPERVOXEL MAP\033[0m");

    //! target object cloud
    for (int i = 0; i < cluster_indices[object_index].indices.size(); i++) {
       int indx = cluster_indices[object_index].indices[i];
       indx = this->region_indices_->indices[indx];
       //! get label of sv
       uint32_t label = indices_map_[indx].label;
       int pt_index = indices_map_[indx].index;
       
       //! prune points and normals of sv
       PointT pt;
       pt.x = std::numeric_limits<float>::quiet_NaN();
       pt.y = std::numeric_limits<float>::quiet_NaN();
       pt.z = std::numeric_limits<float>::quiet_NaN();
       this->supervoxel_clusters_.at(label)->voxels_->points[pt_index] = pt;
    }

    ROS_INFO("\033[36mUPDATING CLOUD\033[0m");
    
    //! update the normals and centroids of the voxels
    // SupervoxelMap supervoxel_copy;
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
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_) \
    shared(merge_list, labels)
#endif
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
                  parent_pt, parent_norm, child_pt, child_norm, -0.025f) == 1) {
              merge_list[i] = index;
              labels[index] = 1;
           } else {
              merge_list[i] = -1;
           }
        } else {
           merge_list[i] = -1;
        }
    }
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_) schedule(guided, 1)
#endif
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
    double distance = 0.0;  // compute this as max distace from center
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
    
    bool is_cluster = true;
    if (is_cluster) {
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
       this->region_indices_->indices.clear();
       for (int i = 0; i < cluster_indices[idx].indices.size(); i++) {
          int pt_indx = cluster_indices[idx].indices[i];
          this->region_indices_->indices.push_back(pt_indx);
       }

       //! update seed point
       int csize = cluster_indices[idx].indices.size() / 2;
       int ind = cluster_indices[idx].indices[csize];
       this->seed_point_ = cloud->points[ind];
       this->seed_normal_ = normals->points[ind];
    }
    
    // TODO(HERE):  if lenght is small merge fix size
    
    ROS_WARN("DISABLE: SEED POINT INFO NOT UPDATED");
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
    euclidean_clustering.setMinClusterSize(this->min_cluster_size_);
    euclidean_clustering.setMaxClusterSize(max_size_thresh);
    euclidean_clustering.setSearchMethod(tree);
    euclidean_clustering.setInputCloud(cloud);
    if (!prob_indices->indices.empty()) {
       euclidean_clustering.setIndices(prob_indices);
    }
    euclidean_clustering.extract(cluster_indices);
}

void ObjectRegionHandler::getRegionSupervoxels(
    SupervoxelMap &region_supervoxels, pcl::PointCloud<PointT>::Ptr region) {
    if (supervoxel_clusters_.empty() || region_indices_->indices.empty()) {
       ROS_ERROR("SUPERVOXEL OR INDEX ARE EMPTY. NOT REGION CREATED");
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
    this->region_indices_->indices.clear();
    for (UInt32Map::iterator it = region_cache.begin();
         it != region_cache.end(); it++) {
       int v_size = supervoxel_clusters_.at(it->first)->voxels_->size() / 2;
       if (it->second > v_size) {
          pcl::Supervoxel<PointT>::Ptr sv = this->supervoxel_clusters_.at(
             it->first);
          pcl::Supervoxel<PointT>::Ptr super_v(new pcl::Supervoxel<PointT>);
          for (int i = 0; i < sv->voxels_->size(); i++) {
             super_v->voxels_->push_back(sv->voxels_->points[i]);
             super_v->normals_->push_back(sv->normals_->points[i]);
          }
          super_v->centroid_ = sv->centroid_;
          super_v->normal_ = sv->normal_;
          region_supervoxels[it->first] = super_v;
          
          *region += *(supervoxel_clusters_.at(it->first)->voxels_);
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
}
