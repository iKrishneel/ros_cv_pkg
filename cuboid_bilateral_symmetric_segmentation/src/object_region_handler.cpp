
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
       pcl::PointIndices::Ptr indices(new pcl::PointIndices);
       int s_size = static_cast<int>(this->in_cloud_->size());
       for (int i = 0; i < it->second->voxels_->size(); i++) {
          indices->indices.push_back(s_size + i);
       }
       this->indices_map_[it->first] = indices;
       *in_cloud_ += *(it->second->voxels_);
       *in_normals_ += *(it->second->normals_);
    }
    this->kdtree_->setInputCloud(this->in_cloud_);
    this->is_init_ = true;
    this->header_ = header;
    this->all_indices_.clear();
    ROS_INFO("\033[34mPOINT CLOUD INFO IS SET FOR SEGMENTATION...\n\033[0m");
    return true;
}

bool ObjectRegionHandler::getCandidateRegion(
    pcl::PointCloud<PointT>::Ptr out_cloud, pcl::PointXYZRGBNormal &info) {
    if (this->in_cloud_->size() < this->min_cluster_size_) {
       ROS_ERROR("INPUT CLOUD NOT SET");
       return false;
    }
    ROS_INFO("\033[34mDOING SUPERVOXEL\033[0m");
    //! do supervoxel

    
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
    
    PointT seed_point;
    if (t_index == supervoxel_clusters_.size() + 1) {
       ROS_ERROR("CANNOT SELECT ANY SUPERVOXEL");
       return false;
    } else {
       this->seed_point_.x = supervoxel_clusters_.at(t_index)->centroid_.x;
       this->seed_point_.y = supervoxel_clusters_.at(t_index)->centroid_.y;
       this->seed_point_.z = supervoxel_clusters_.at(t_index)->centroid_.z;
       this->seed_point_.r = supervoxel_clusters_.at(t_index)->centroid_.r;
       this->seed_point_.g = supervoxel_clusters_.at(t_index)->centroid_.g;
       this->seed_point_.b = supervoxel_clusters_.at(t_index)->centroid_.b;
       this->seed_normal_ = supervoxel_clusters_.at(t_index)->normal_;

       // ? NO NEED
       /*
         int k = 1;
       std::vector<int> neigbor_indices;
       this->pointNeigbour<int>(neigbor_indices, point, k);
       int idx = neigbor_indices[0];
       Eigen::Vector4f cpt = point.getVector4fMap();
       Eigen::Vector4f npt = in_cloud_->points[idx].getVector4fMap();
       cpt(3) = 1.0f;
       npt(3) = 1.0f;
       double d = pcl::distances::l2(cpt, npt);
       if (d < this->voxel_resolution_ * 2.0) {  // ?? CONFIRM THIS
          this->seed_index_ = idx;
          seed_point = this->in_cloud_->points[idx];
       }
       */
    }
    
    ROS_INFO("\033[34mDOING REGION GROWING\033[0m");
    
    //! region growing
    // pcl::PointCloud<NormalT>::Ptr normals(new pcl::PointCloud<NormalT>);
    // this->estimateNormals<int>(this->in_cloud_, normals, neigbor_size_;
    
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

          this->region_indices_->indices.push_back(i);
       }
    }

    std::cout << "SEED_SIZE: " << seed_cloud->size() << "\t"
              << seed_normals->size()  << "\n";
    
    //! get lenght
    this->regionOverSegmentation(seed_cloud, seed_normals,
                                 this->in_cloud_, this->in_normals_);
    out_cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*seed_cloud, *out_cloud);
    
    info.x = this->seed_point_.x;
    info.y = this->seed_point_.y;
    info.z = this->seed_point_.z;
    info.r = this->seed_point_.r;
    info.b = this->seed_point_.b;
    info.g = this->seed_point_.g;
    info.normal_x = this->seed_normal_.normal_x;
    info.normal_y = this->seed_normal_.normal_x;
    info.normal_z = this->seed_normal_.normal_x;
    info.curvature = this->seed_normal_.normal_x;

    ROS_INFO("\033[34mDONE: %d\033[0m", iter_counter_);

    if (iter_counter_-- == 0) {
       return false;
    }
    return true;
}

void ObjectRegionHandler::updateObjectRegion(
    pcl::PointCloud<PointT>::Ptr cloud) {
    if (cloud->empty()) {
       return;
    }

    ROS_INFO("\033[36mDOING CLUSTERING\033[0m");
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::PointIndices::Ptr prob_indices(new pcl::PointIndices);
    this->doEuclideanClustering(cluster_indices, cloud, prob_indices);

    if (cluster_indices.empty()) {
       return;
    }

    ROS_INFO("\033[36mGETTING THE BEST CLOUD\033[0m");
    
    //! get the region
    PointT seed_point = this->seed_point_;
    double distance = DBL_MAX;
    for (int i = 0; i < cluster_indices.size(); i++) {
       pcl::PointCloud<PointT>::Ptr temp(new pcl::PointCloud<PointT>);
       for (int j = 0; j < cluster_indices[i].indices.size(); j++) {
          int idx = cluster_indices[i].indices[j];
          temp->push_back(cloud->points[idx]);
       }
       
       Eigen::Vector4f centroid;
       pcl::compute3DCentroid<PointT, float>(*temp, centroid);
       double d = pcl::distances::l2(centroid, seed_point.getVector4fMap());
       if (d < distance) {
          distance = d;
          cloud->clear();
          pcl::copyPointCloud<PointT, PointT>(*temp, *cloud);
       }
    }

    ROS_INFO("\033[36mUPDATING CLOUD\033[0m");
    
    std::vector<int> neigbor_indices;
    pcl::PointIndices indices;
    for (int i = 0; i < cloud->size(); i++) {
       neigbor_indices.clear();
       this->pointNeigbour<float>(neigbor_indices, cloud->points[i],
                                  this->voxel_resolution_, false);
       for (int j = 0; j < neigbor_indices.size(); j++) {
          int idx = neigbor_indices[j];
          in_cloud_->points[idx].x = std::numeric_limits<double>::quiet_NaN();
          in_cloud_->points[idx].y = std::numeric_limits<double>::quiet_NaN();
          in_cloud_->points[idx].z = std::numeric_limits<double>::quiet_NaN();
          indices.indices.push_back(idx);
       }
    }
    this->all_indices_.push_back(indices);
    
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
           
           // Eigen::Vector4f parent_pt = cloud->points[
           //    this->seed_index_].getVector4fMap();
           // Eigen::Vector4f parent_norm = normals->points[
           //    this->seed_index_].getNormalVector4fMap();
            
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
    if (seed2pt_relation > thresh &&
        pt2seed_relation > thresh && norm_similarity > 0.50f) {
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
       ROS_ERROR("THE CLOUD IS EMPTY. RETURING VOID IN GET NEIGBOUR");
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

    double dist = DBL_MAX;
    int idx = -1;
    int icount = 0;
    pcl::PointIndices::Ptr prob_indices(new pcl::PointIndices);
    PointT seed_point = this->seed_point_;
    for (int i = 0; i < cloud->size(); i++) {
       if (pcl::distances::l2(cloud->points[i].getVector4fMap(),
                              center) < distance) {
          PointT pt = cloud->points[i];
          if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z)) {
             region->push_back(cloud->points[i]);
             normal->push_back(normals->points[i]);
             prob_indices->indices.push_back(i);
             
             // update the seed info
             double d = pcl::distances::l2(cloud->points[i].getVector4fMap(),
                                           seed_point.getVector4fMap());
             if (d < dist) {
                dist = d;
                idx = icount;
             }
             icount++;
          }
       }
    }
    // pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    std::vector<pcl::PointIndices> cluster_indices;
    cluster_indices.clear();
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);
    pcl::EuclideanClusterExtraction<PointT> euclidean_clustering;
    euclidean_clustering.setClusterTolerance(0.01f);
    euclidean_clustering.setMinClusterSize(this->min_cluster_size_);
    euclidean_clustering.setMaxClusterSize(25000);
    euclidean_clustering.setSearchMethod(tree);
    euclidean_clustering.setInputCloud(cloud);
    euclidean_clustering.setIndices(prob_indices);
    euclidean_clustering.extract(cluster_indices);
    double min_distance = DBL_MAX;
    idx = -1;
    dist = DBL_MAX;
    for (int i = 0; i < cluster_indices.size(); i++) {
       pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
       pcl::PointCloud<NormalT>::Ptr tmp_normal(new pcl::PointCloud<NormalT>);
       for (int j = 0; j < cluster_indices[i].indices.size(); j++) {
          int cidx = cluster_indices[i].indices[j];
          tmp_cloud->push_back(cloud->points[cidx]);
          tmp_normal->push_back(normals->points[cidx]);
       }
       
       Eigen::Vector4f center;
       pcl::compute3DCentroid<PointT, float>(*tmp_cloud, center);
       double d = pcl::distances::l2(seed_point.getVector4fMap(), center);
       if (d < dist) {
          dist = d;
          idx = i;

          region->clear();
          pcl::copyPointCloud<PointT, PointT>(*tmp_cloud, *region);
          normal->clear();
          pcl::copyPointCloud<NormalT, NormalT>(*tmp_normal, *normal);
       }
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

