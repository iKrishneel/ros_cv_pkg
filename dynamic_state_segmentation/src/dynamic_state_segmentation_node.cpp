
#include <dynamic_state_segmentation/dynamic_state_segmentation.h>

DynamicStateSegmentation::DynamicStateSegmentation() :
    num_threads_(16), neigbor_size_(50) {
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    this->onInit();
}

void DynamicStateSegmentation::onInit() {
    this->srv_client_ = this->pnh_.serviceClient<
       dynamic_state_segmentation::Feature3DClustering>(
          "feature3d_clustering_srv");
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "target", 1);
    this->pub_edge_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "edge", 1);
     this->pub_indices_ = this->pnh_.advertise<
        jsk_recognition_msgs::ClusterPointIndices>("indices", 1);
}

void DynamicStateSegmentation::subscribe() {
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->screen_pt_.subscribe(this->pnh_, "input_point", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
                                       SyncPolicy> >(100);
    this->sync_->connectInput(sub_cloud_, screen_pt_);
    this->sync_->registerCallback(
        boost::bind(&DynamicStateSegmentation::cloudCB, this, _1, _2));
}

void DynamicStateSegmentation::unsubscribe() {
    this->sub_cloud_.unsubscribe();
}

void DynamicStateSegmentation::cloudCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const geometry_msgs::PointStamped::ConstPtr &screen_msg) {
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    std::cout << "INPUT SIZE: " << cloud->size()  << "\n";
    
    if (cloud->empty()) {
        return;
    }

    this->seed_index_ = screen_msg->point.x + (640 * screen_msg->point.y);
    this->seed_point_ = cloud->points[seed_index_];

    if (isnan(this->seed_point_.x) || isnan(this->seed_point_.x) ||
        isnan(this->seed_point_.x)) {
        ROS_ERROR("SELETED POINT IS NAN");
        return;
    }

    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud<PointT>(*cloud, *cloud, nan_indices);

    std::cout << "INPUT FILTERED SIZE: " << cloud->size()  << "\n";
    
    // temp update the seed based on removal shift
    double dist = DBL_MAX;
    int idx = -1;
    for (int i = 0; i < cloud->size(); i++) {
        double d = pcl::distances::l2(cloud->points[i].getVector4fMap(),
                                      seed_point_.getVector4fMap());
        if (d < dist) {
            dist = d;
            idx = i;
        }
    }
    
    ROS_INFO("PROCESSING");
    
    std::vector<int> labels(static_cast<int>(cloud->size()));
    for (int i = 0; i < cloud->size(); i++) {
        if (i == this->seed_index_) {
            labels[i] = 1;
        }
        labels[i] = -1;
    }

    pcl::PointCloud<NormalT>::Ptr normals(new pcl::PointCloud<NormalT>);
    this->estimateNormals<int>(cloud, normals, this->neigbor_size_, true);
    this->seed_index_ = idx;
    this->seed_point_ = cloud->points[idx];
    this->seed_normal_ = normals->points[idx];

    kdtree_->setInputCloud(cloud);

    ROS_INFO("GROWING SEED");
    this->seedCorrespondingRegion(labels, cloud, normals, this->seed_index_);

    pcl::PointCloud<PointT>::Ptr seed_cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<NormalT>::Ptr seed_normals(new pcl::PointCloud<NormalT>);
    pcl::PointIndices a_indices;
    for (int i = 0; i < labels.size(); i++) {
        if (labels[i] != -1) {
           PointT pt = cloud->points[i];
           seed_cloud->push_back(pt);
           seed_normals->push_back(normals->points[i]);
           a_indices.indices.push_back(i);
        }
    }
    std::vector<pcl::PointIndices> all_indices;
    all_indices.push_back(a_indices);
    
    ROS_INFO("DONE.");

    // process estimated cloud and update seed point info
    this->regionOverSegmentation(seed_cloud, seed_normals, cloud, normals);

    pcl::PointCloud<PointT>::Ptr convex(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr concave(new pcl::PointCloud<PointT>);
    this->normalEdge(convex, concave, seed_cloud, seed_normals);

    cloud->clear();
    *cloud = *concave;
    
    this->kdtree_->setInputCloud(seed_cloud);
    this->dynamicSegmentation(cloud, seed_cloud, seed_normals);

    
    // cloud->clear();
    // seed_cloud->clear();
    // *cloud = *convex;
    // *seed_cloud = *concave;
    
    /*
    std::vector<std::vector<int> > neigbor_indices;
    pcl::PointCloud<PointT>::Ptr appearance_weights(new pcl::PointCloud<PointT>);
    this->potentialFunctionKernel(neigbor_indices, appearance_weights,
				  seed_cloud, seed_normals);
    seed_cloud->clear();
    *seed_cloud = *appearance_weights;
    */
    
    jsk_recognition_msgs::ClusterPointIndices ros_indices;
    ros_indices.cluster_indices = pcl_conversions::convertToROSPointIndices(
       all_indices, cloud_msg->header);
    ros_indices.header = cloud_msg->header;
    this->pub_indices_.publish(ros_indices);

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);

    sensor_msgs::PointCloud2 ros_edge;
    pcl::toROSMsg(*seed_cloud, ros_edge);
    ros_edge.header = cloud_msg->header;
    this->pub_edge_.publish(ros_edge);
}

void DynamicStateSegmentation::seedCorrespondingRegion(
    std::vector<int> &labels, const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<NormalT>::Ptr normals, const int parent_index) {
    std::vector<int> neigbor_indices;
    this->getPointNeigbour<int>(neigbor_indices, cloud,
                                cloud->points[parent_index],
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
            Eigen::Vector4f parent_pt = cloud->points[
                parent_index].getVector4fMap();
            Eigen::Vector4f parent_norm = normals->points[
                parent_index].getNormalVector4fMap();
            Eigen::Vector4f child_pt = cloud->points[index].getVector4fMap();
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

template<class T>
void DynamicStateSegmentation::getPointNeigbour(
    std::vector<int> &neigbor_indices, const pcl::PointCloud<PointT>::Ptr cloud,
    const PointT seed_point, const T K, bool is_knn) {
    if (cloud->empty() || isnan(seed_point.x) ||
        isnan(seed_point.y) || isnan(seed_point.z)) {
        ROS_ERROR("THE CLOUD IS EMPTY. RETURING VOID IN GET NEIGBOUR");
        return;
    }
    neigbor_indices.clear();
    std::vector<float> point_squared_distance;
    if (is_knn) {
        int search_out = kdtree_->nearestKSearch(
            seed_point, K, neigbor_indices, point_squared_distance);
    } else {
        int search_out = kdtree_->radiusSearch(
            seed_point, K, neigbor_indices, point_squared_distance);
    }
}

int DynamicStateSegmentation::seedVoxelConvexityCriteria(
    Eigen::Vector4f c_centroid, Eigen::Vector4f c_normal,
    Eigen::Vector4f n_centroid, Eigen::Vector4f n_normal,
    const float thresh, bool is_seed) {
    float im_relation = (n_centroid - c_centroid).dot(n_normal);
    float pt2seed_relation = FLT_MAX;
    float seed2pt_relation = FLT_MAX;
    if (is_seed) {
       pt2seed_relation = (n_centroid -
                           this->seed_point_.getVector4fMap()).dot(n_normal);
       seed2pt_relation = (this->seed_point_.getVector4fMap() - n_centroid).dot(
          this->seed_normal_.getNormalVector4fMap());
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
void DynamicStateSegmentation::estimateNormals(
    const pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<NormalT>::Ptr normals, const T k, bool use_knn) const {
    if (cloud->empty()) {
        ROS_ERROR("ERROR: The Input cloud is Empty.....");
        return;
    }
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


void DynamicStateSegmentation::regionOverSegmentation(
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
    for (int i = 0; i < cloud->size(); i++) {
       if (pcl::distances::l2(cloud->points[i].getVector4fMap(),
                              center) < distance) {
          region->push_back(cloud->points[i]);
          normal->push_back(normals->points[i]);

	  // update the seed info
	  double d = pcl::distances::l2(cloud->points[i].getVector4fMap(),
					this->seed_point_.getVector4fMap());
	  if (d < dist) {
	      dist = d;
	      idx = icount;
	  }
	  icount++;
       }
    }
    if (idx != -1 && icount == region->size()) {
	this->seed_index_ = idx;
	this->seed_point_ = region->points[idx];
	this->seed_normal_ = normal->points[idx];
    } else {
	ROS_WARN("SEED POINT INFO NOT UPDATED");
    }
}

/**
 * smoothness term
 */

int DynamicStateSegmentation::localVoxelConvexityCriteria(
    Eigen::Vector4f c_centroid, Eigen::Vector4f n_centroid,
    Eigen::Vector4f n_normal, const float threshold) {
    if ((n_centroid - c_centroid).dot(n_normal) > 0) {
       return 1;
    } else {
       return -1;
    }
}

void DynamicStateSegmentation::normalEdge(
    pcl::PointCloud<PointT>::Ptr convex_edge_pts,
    pcl::PointCloud<PointT>::Ptr concave_edge_pts,
    const pcl::PointCloud<PointT>::Ptr in_cloud,
    const pcl::PointCloud<NormalT>::Ptr in_normals) {
    if (in_cloud->empty() || in_cloud->size() != in_normals->size()) {
       ROS_ERROR("INCORRECT POINT SIZE FOR EDGE FUNCTION");
       return;
    }
    convex_edge_pts->resize(static_cast<int>(in_cloud->size()));
    concave_edge_pts->resize(static_cast<int>(in_cloud->size()));
    ROS_INFO("COMPUTING EDGE");
    
    pcl::KdTreeFLANN<PointT>::Ptr kdtree(new pcl::KdTreeFLANN<PointT>());
    kdtree->setInputCloud(in_cloud);
    const float radius_thresh = 0.01f;
    const float concave_thresh = 0.25f;
    std::vector<int> point_idx_search;
    std::vector<float> point_squared_distance;
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_) shared(kdtree) \
    private(point_idx_search, point_squared_distance)
#endif
    for (int i = 0; i < in_cloud->size(); i++) {
       PointT centroid_pt = in_cloud->points[i];
       if (!isnan(centroid_pt.x) || !isnan(centroid_pt.y) ||
           !isnan(centroid_pt.z)) {
          point_idx_search.clear();
          point_squared_distance.clear();
          int search_out = kdtree->radiusSearch(
             centroid_pt, radius_thresh, point_idx_search,
             point_squared_distance);
          Eigen::Vector4f seed_vector = in_normals->points[
             i].getNormalVector4fMap();
          float max_diff = 0.0f;
          float min_diff = FLT_MAX;

          int concave_sum = 0;
          for (int j = 1; j < point_idx_search.size(); j++) {
             int index = point_idx_search[j];
             Eigen::Vector4f neigh_norm = in_normals->points[
                index].getNormalVector4fMap();
             float dot_prod = neigh_norm.dot(seed_vector) / (
                neigh_norm.norm() * seed_vector.norm());
             
             if (dot_prod > max_diff) {
                max_diff = dot_prod;
             }
             if (dot_prod < min_diff) {
                min_diff = dot_prod;
             }

             concave_sum += this->localVoxelConvexityCriteria(
                centroid_pt.getVector4fMap(), in_cloud->points[
                   index].getVector4fMap(), neigh_norm);
          }
          float variance = max_diff - min_diff;

          if (variance > concave_thresh && concave_sum <= 0) {
             centroid_pt.r = 255 * variance;
             centroid_pt.b = 0;
             centroid_pt.g = 0;
             concave_edge_pts->points[i] = centroid_pt;
          }
          if (variance > 0.20f && concave_sum > 0) {
             centroid_pt.g = 255 * variance;
             centroid_pt.b = 0;
             centroid_pt.r = 0;
             convex_edge_pts->points[i] = centroid_pt;
          }
       }
    }
    // filter the outliers
#ifdef _OPENMP
#pragma omp parallel sections
#endif
    {

#ifdef _OPENMP
#pragma omp section
#endif
	{
	    // double ccof = this->outlier_concave_
	    pcl::PointCloud<PointT>::Ptr temp_pt(new pcl::PointCloud<PointT>);
	    pcl::copyPointCloud<PointT, PointT>(*concave_edge_pts, *temp_pt);
	    double ccof = 0.015;
	    pcl::PointIndices::Ptr cc_indices(new pcl::PointIndices);
	    this->edgeBoundaryOutlierFiltering(temp_pt, cc_indices,
					       static_cast<float>(ccof));
	    for (int i = 0; i < cc_indices->indices.size(); i++) {
		int index = cc_indices->indices[i];
		concave_edge_pts->points[index].r = 0;
	    }

	}

#ifdef _OPENMP
#pragma omp section
#endif
	{
	    // double cvof = this->outlier_convex_;
	    pcl::PointCloud<PointT>::Ptr temp_pt(new pcl::PointCloud<PointT>);
	    pcl::copyPointCloud<PointT, PointT>(*convex_edge_pts, *temp_pt);
	    double cvof = 0.015;
	    pcl::PointIndices::Ptr cv_indices(new pcl::PointIndices);
	    this->edgeBoundaryOutlierFiltering(temp_pt, cv_indices,
					       static_cast<float>(cvof));
	    for (int i = 0; i < cv_indices->indices.size(); i++) {
		int index = cv_indices->indices[i];
		convex_edge_pts->points[index].g = 0;
	    }
	}
    }

    std::vector<int>(point_idx_search).swap(point_idx_search);
    std::vector<float>(point_squared_distance).swap(point_squared_distance);
    
    ROS_INFO("EDGE COMPUTED");
}

void DynamicStateSegmentation::edgeBoundaryOutlierFiltering(
    pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr removed_indices,
    const float search_radius_thresh, const int min_neigbor_thresh) {
    if (cloud->empty()) {
	ROS_WARN("SKIPPING OUTLIER FILTERING");
	return;
    }
    ROS_INFO("\033[32m FILTERING OUTLIER \033[0m");
    pcl::PointCloud<PointT>::Ptr concave_edge_points(
	new pcl::PointCloud<PointT>);
    pcl::RadiusOutlierRemoval<PointT>::Ptr filter_ror(
	new pcl::RadiusOutlierRemoval<PointT>(true));
    filter_ror->setInputCloud(cloud);
    filter_ror->setRadiusSearch(search_radius_thresh);
    filter_ror->setMinNeighborsInRadius(min_neigbor_thresh);
    filter_ror->filter(*concave_edge_points);
    cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*concave_edge_points, *cloud);
    filter_ror->getRemovedIndices(*removed_indices);
}

template<class T>
T DynamicStateSegmentation::distancel2(
    const Eigen::Vector3f point1, const Eigen::Vector3f point2, bool is_root) {
    T distance = std::pow(point1(0) - point2(0), 2) +
       std::pow(point1(1) - point2(1), 2) + std::pow(point1(2) - point2(2), 2);
    if (is_root) {
       distance = std::sqrt(distance);
    }
    return distance;
}

void DynamicStateSegmentation::potentialFunctionKernel(
    std::vector<std::vector<int> > &neigbor_cache,
    pcl::PointCloud<PointT>::Ptr weights,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<NormalT>::Ptr normals) {
    if (cloud->empty() || cloud->size() != normals->size()) {
       ROS_ERROR("INAPPOPRIATE SIZE");
       return;
    }

    neigbor_cache.clear();
    neigbor_cache.resize(static_cast<int>(cloud->size()));
    
    pcl::copyPointCloud<PointT, PointT>(*cloud, *weights);
    const float weight_param = 1.0f;
    const float position_param = 1.0f;
    const float color_param = 1.0f;
    for (int i = 0; i < cloud->size(); i++) {
       std::vector<int> neigbor_indices;
       this->getPointNeigbour<int>(neigbor_indices, cloud, cloud->points[i], 8);
       neigbor_cache[i] = neigbor_indices;
	
       float sum = 0.0f;
       Eigen::Vector3f center_pt = cloud->points[i].getVector3fMap();
       for (int j = 0; j < neigbor_indices.size(); j++) {
          float convex_term = (cloud->points[j].getVector3fMap() -
                               center_pt).dot(normals->points[
                                                 j].getNormalVector3fMap());
          
          /*
	    int convex_term = this->seedVoxelConvexityCriteria(
            cloud->points[i].getVector4fMap(),
            normals->points[i].getNormalVector4fMap(),
            cloud->points[j].getVector4fMap(),
            normals->points[j].getNormalVector4fMap(),
            -0.02f);
          */
          float norm_dif = normals->points[i].getNormalVector4fMap().dot(
             normals->points[j].getNormalVector4fMap()) / (
                normals->points[i].getNormalVector4fMap().norm() *
                normals->points[j].getNormalVector4fMap().norm());
          float v_term = 0.0f;
          if (convex_term > 0.0) {
             v_term = std::exp(-norm_dif/(2 * M_PI));
          } else {
             v_term = std::exp(-norm_dif/(M_PI/3));
          }

          float curvature1 = normals->points[i].curvature;
          float curvature2 = normals->points[j].curvature;
	    
          // sum += v_term;
          // sum += (std::exp(-1 * (curvature1/curvature2)));
          float curvature = (((curvature1 - curvature2) * (
                                 curvature1 - curvature2)) / (2.0f * 1.0));
          sum += (0.70 * std::exp(- curvature) + (1.0f * v_term));
	    
          float pos_dif = this->distancel2<float>(
             cloud->points[j].getVector3fMap(), center_pt, true);
          float position = (pos_dif * pos_dif) / (
             2.0f * position_param * position_param);

          float col_dif = this->intensitySimilarityMetric<float>(
             cloud->points[i], cloud->points[j], true);
	    
          norm_dif = std::acos(norm_dif);
          // float col_dif = norm_dif;
	    
          float intensity = (col_dif * col_dif) / (
             2.0f * color_param * color_param);
          float app_kernel = std::exp(-position - intensity) * 0.5f;
	    
          // smoothness
          float smoothness = weight_param * std::exp(-position);
	    
          // sum += (app_kernel + smoothness);
       }

       // float avg = sum / static_cast<float>(neigbor_indices.size());
       float avg = sum;

       float pixel = 255.0f;
       weights->points[i].r = (avg * pixel);
       weights->points[i].g = (avg * pixel);
       weights->points[i].b = (avg * pixel);
    }
}


/**
 * energy function
 */
void DynamicStateSegmentation::dynamicSegmentation(
    pcl::PointCloud<PointT>::Ptr region, pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<NormalT>::Ptr normals) {
    if (cloud->size() != normals->size() || cloud->empty()) {
       ROS_ERROR("INCORRECT SIZE FOR DYNAMIC STATE");
       return;
    }
    
    ROS_INFO("\033[34m GRAPH CUT \033[0m");
    
    const int node_num = cloud->size();
    const int edge_num = 8;
    boost::shared_ptr<GraphType> graph(new GraphType(
                                          node_num, edge_num * node_num));
    
    for (int i = 0; i < node_num; i++) {
       graph->add_node();
    }

    ROS_INFO("\033[34m GRAPH INIT %d \033[0m", node_num);
    
    pcl::PointCloud<PointT>::Ptr weight_map(new pcl::PointCloud<PointT>);
    std::vector<std::vector<int> > neigbor_indices;
    this->potentialFunctionKernel(neigbor_indices, weight_map, cloud, normals);

    ROS_INFO("\033[34m POTENTIAL COMPUTED \033[0m");

    // get seed_region neigbours for hard label
    const float s_radius = 0.02f;
    std::vector<int> seed_neigbors;
    this->getPointNeigbour<float>(seed_neigbors, cloud, this->seed_point_, s_radius, false);

    std::vector<bool> label_cache(static_cast<int>(cloud->size()));
    for (int i = 0; i < cloud->size(); i++) {
	label_cache[i] = false;
    }

    for (int i = 0; i < seed_neigbors.size(); i++) {
	graph->add_tweights(i, HARD_THRESH, 0);
	label_cache[i] = true;
    }
    
    for (int i = 0; i < weight_map->size(); i++) {
       float weight = weight_map->points[i].r;
       float edge_weight = region->points[i].r;
       // if (weight > obj_thresh) {
       // 	   graph->add_tweights(i, HARD_THRESH, 0);
       // } else
       if (/*weight < bkgd_thresh*/ edge_weight > 0) {
	   graph->add_tweights(i, 0, HARD_THRESH);
       } else if (!label_cache[i]){
	   float w = -std::log(weight/255.0) * 10;
	   if (weight == 0) {
	       w = -std::log(1e-9);
	   }
	   // if (isnan(w)) {
	   // }
	   // std::cout<< "\t" << w << "\t" << weight<< "\n";
	   graph->add_tweights(i, w, w);
       }
	
       for (int j = 0; j < neigbor_indices[i].size(); j++) {
          int indx = neigbor_indices[i][j];
          if (indx != i) {
	      // float w = std::pow(weight - weight_map->points[indx].r, 2);
	      float r = std::abs(cloud->points[indx].r - cloud->points[i].r);
	      float g = std::abs(cloud->points[indx].g - cloud->points[i].g);
	      float b = std::abs(cloud->points[indx].b - cloud->points[i].b);
	      float w = (r + g + b) /255.0f;
	      
	      // w = std::sqrt(w);
	      w = 1/w;
	      if (isnan(w)) {
		  w = FLT_MIN;
	      }
	      // std::cout << w  << ", ";
	      graph->add_edge(i, indx, (w), (w));
          }
       }
    }
    
    
    ROS_INFO("\033[34m COMPUTING FLOW \033[0m");
    
    float flow = graph->maxflow();

    ROS_INFO("\033[34m FLOW: %3.2f \033[0m", flow);
    // plot
    region->clear();
    for (int i = 0; i < node_num; i++) {
       if (graph->what_segment(i) == GraphType::SOURCE) {
          region->push_back(cloud->points[i]);
       } else {
          continue;
       }
    }
    ROS_INFO("\033[34m DONE: %d \033[0m", region->size());

    cloud->clear();
    *cloud = *weight_map;
}


/**
 * 
 */

void DynamicStateSegmentation::clusterFeatures(
    std::vector<pcl::PointIndices> &all_indices,
    pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<NormalT>::Ptr descriptors,
    const int min_size, const float max_distance) {
    if (descriptors->empty() || descriptors->size() != cloud->size()) {
       ROS_ERROR("ERROR: EMPTY FEATURES.. SKIPPING CLUSTER SRV");
       return;
    }
    dynamic_state_segmentation::Feature3DClustering srv;
    std::vector<std::vector<int> > neigbor_cache(
       static_cast<int>(descriptors->size()));
    for (int i = 0; i < descriptors->size(); i++) {
       jsk_recognition_msgs::Histogram hist;
       hist.histogram.push_back(cloud->points[i].x);
       hist.histogram.push_back(cloud->points[i].y);
       hist.histogram.push_back(cloud->points[i].z);
	
	// hist.histogram.push_back(descriptors->points[i].normal_x);
	// hist.histogram.push_back(descriptors->points[i].normal_y);
	// hist.histogram.push_back(descriptors->points[i].normal_z);

       std::vector<int> neigbor_indices;
       this->getPointNeigbour<int>(neigbor_indices, cloud, cloud->points[i], 8);
       neigbor_cache[i] = neigbor_indices;
       
       float sum = 0.0f;
       Eigen::Vector4f c_n = descriptors->points[i].getNormalVector4fMap();

       // planner
       Eigen::Vector4f c_pt = cloud->points[i].getVector4fMap();
       float coplanar = 0.0f;
       float curvature = 0.0f;

       for (int j = 0; j < neigbor_indices.size(); j++) {
          int idx = neigbor_indices[j];
          Eigen::Vector4f n_n = descriptors->points[idx].getNormalVector4fMap();
          sum += (c_n.dot(n_n));
          Eigen::Vector4f n_pt = cloud->points[idx].getVector4fMap();
          coplanar += ((n_pt - c_pt).dot(n_n));
          curvature += descriptors->points[idx].curvature;
       }
       hist.histogram.push_back(sum/static_cast<float>(neigbor_indices.size()));
       // hist.histogram.push_back(coplanar /
       //                          static_cast<float>(neigbor_indices.size()));

       hist.histogram.push_back(curvature/static_cast<float>(
                                   neigbor_indices.size()));
       srv.request.features.push_back(hist);
    }
    srv.request.min_samples = min_size;
    srv.request.max_distance = max_distance;
    if (this->srv_client_.call(srv)) {
       int max_label = srv.response.argmax_label;
       if (max_label == -1) {
          return;
       }
       all_indices.clear();
       all_indices.resize(max_label + 20);
       for (int i = 0; i < srv.response.labels.size(); i++) {
          int index = srv.response.labels[i];
          if (index > -1) {
             all_indices[index].indices.push_back(i);
          }
       }


       this->mergeVoxelClusters(srv, cloud, descriptors, neigbor_cache);
	
	
    } else {
       ROS_ERROR("SRV CLIENT CALL FAILED");
       return;
    }

    ROS_WARN("CLUSTERED.");
}


void DynamicStateSegmentation::mergeVoxelClusters(
    // std::vector<pcl::PointIndices> &all_indices,
    const dynamic_state_segmentation::Feature3DClustering srv,
    pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<NormalT>::Ptr normals,
    const std::vector<std::vector<int> > neigbor_cache) {
    pcl::PointCloud<PointT>::Ptr cluster_points(new pcl::PointCloud<PointT>);

    std::vector<int> boundary_indices;
    std::vector<std::vector<int> > boundary_adj(20);
    for (int i = 0; i < srv.response.labels.size(); i++) {
       int label = srv.response.labels[i];
       if (srv.response.labels[i] != -1) {
          int idx = neigbor_cache[i][0];
          int lab = srv.response.labels[idx];
          bool is_neigb = false;
          for (int j = 1; j < neigbor_cache[i].size(); j++) {
             idx = neigbor_cache[i][j];
             if (srv.response.labels[idx] != -1) {
                if (lab != srv.response.labels[idx]) {
                   is_neigb = true;
                   cluster_points->push_back(cloud->points[i]);

                   boundary_indices.push_back(i);
                   boundary_adj[label].push_back(srv.response.labels[idx]);
                }
             }
             if (is_neigb) {
                j += FLT_MAX;
             }
          }
       }
    }

    for (int i = 0; i < boundary_adj.size(); i++) {
       std::vector<int> neigbour_labels = boundary_adj[i];
       if (!neigbour_labels.empty()) {
          std::sort(neigbour_labels.begin(), neigbour_labels.end(), sortVector);
       }
       std::cout << "\n MAIN: " << i  << "\n";

       // find tthe closes point on this cluster
	
       for (int j = 0; j < neigbour_labels.size(); j++) {
	    
          std::cout << neigbour_labels[j]  << ", ";
       }
       std::cout << "\n";
    }
    
    cloud->clear();
    *cloud = *cluster_points;
}


/**
 * --------------------------------------------------------------------------------
 */
void DynamicStateSegmentation::computeFeatures(
    pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<NormalT>::Ptr normals, const int index1) {
    if (cloud->empty() || normals->empty() ||
        cloud->size() != normals->size()) {
        ROS_ERROR("THE CLOUD IS EMPTY. RETURING VOID IN FEATURES");
        return;
    }
    this->kdtree_->setInputCloud(cloud);
    // compute normal
    pcl::PointCloud<NormalT>::Ptr seed_normals(new pcl::PointCloud<NormalT>);
    // this->estimateNormals<float>(cloud, seed_normals, 0.05f, false);
    
    /**
     * NORMAL
     */
    int level = 0;
    int levels = 4;
    float search_radius = 0.04f;
    int search_k = 8;
    
    float max_sum = 0;
    
    do {
       for (int i = 0; i < cloud->size(); i++) {
          std::vector<int> neigbor_indices;

          this->getPointNeigbour<float>(neigbor_indices, cloud,
                                        cloud->points[i], search_radius, false);
           // this->getPointNeigbour<int>(neigbor_indices, cloud,
           //                             cloud->points[i], search_k, true);
	    
	
          float sum = 0.0f;
          Eigen::Vector3f c_pt = normals->points[i].getNormalVector3fMap();
          Eigen::Vector3f n_pt;
          for (int j = 1; j < neigbor_indices.size(); j++) {
             int index = neigbor_indices[j];
             n_pt = normals->points[index].getNormalVector3fMap();
             sum += (c_pt.dot(n_pt));
          }
          // sum /= static_cast<float>(neigbor_indices.size() - 1);

          if (sum > max_sum) {
             max_sum = sum;
          }
	    
          if (level == 0) {
             cloud->points[i].r = sum;
             cloud->points[i].b = sum;
             cloud->points[i].g = sum;
          } else {
             cloud->points[i].r += sum;
             cloud->points[i].b += sum;
             cloud->points[i].g += sum;
          }
	    
          if (level + 1 == levels) {
             max_sum = 1;
             cloud->points[i].r = (cloud->points[i].r/ max_sum)  * 255.0f;
             cloud->points[i].b = (cloud->points[i].b/ max_sum)  * 255.0f;
             cloud->points[i].g = (cloud->points[i].g/ max_sum)  * 255.0f;
          }
       }
       search_radius /= 2.0f;
       search_k *= 2;

       std::cout << search_k << "\n";
	
    } while (level++ < levels);

    
    return;

    /**
     * FPFH
     */
    
    // save the compute neigbour
    std::vector<std::vector<int> > cache_neigbour(static_cast<int>(cloud->size()));
    const int knn = 8;
    
    pcl::PointCloud<FPFH>::Ptr fpfhs (new pcl::PointCloud<FPFH>());
    this->computeFPFH(fpfhs, cloud, normals, 0.03f);

    // FPFH weight
    std::vector<float> fpfh_weight(static_cast<int>(cloud->size()));
    float max_hist_dist = 0.0f;
    for (int i = 0; i < fpfhs->size(); i++) {
       this->getPointNeigbour<int>(cache_neigbour[i], cloud,
                                   cloud->points[i], knn);
       float inter_dist = 0.0f;
       int icount = 0;
       for (int k = 1; k < cache_neigbour[i].size(); k++) {
          int index = cache_neigbour[i][k];
          inter_dist += this->histogramKernel<float>(fpfhs->points[i],
                                                     fpfhs->points[index], 33);
          icount++;
       }

       // std::cout << inter_dist  << "\n";
	
       fpfh_weight[i] = inter_dist/static_cast<float>(icount);
       // fpfh_weight[i] = inter_dist;
       if (fpfh_weight[i] > max_hist_dist) {
          max_hist_dist = fpfh_weight[i];
       }
    }

    std::cout << "MAX DIST:  " << max_hist_dist  << "\n";

    max_hist_dist = 1;
    
    for (int i = 0; i < fpfh_weight.size(); i++) {
       PointT pt = cloud->points[i];
       pt.r = (fpfh_weight[i] / max_hist_dist) * 255.0;
       pt.b = (fpfh_weight[i] / max_hist_dist) * 255.0;
       pt.g = (fpfh_weight[i] / max_hist_dist) * 255.0;
       cloud->points[i] = pt;
    }

}

template<class T>
T DynamicStateSegmentation::histogramKernel(
    const FPFH histA, const FPFH histB, const int bin_size) {
    T sum = 0.0;
    for (int i = 0; i < bin_size; i++) {
       float a = isnan(histA.histogram[i]) ? 0.0 : histA.histogram[i];
       float b = isnan(histB.histogram[i]) ? 0.0 : histB.histogram[i];
       sum += (a + b - std::abs(a - b));
       // sum += (std::min(a, b));
    }
    return (0.5 * sum);
}

/**
 * DESCRIPTORS
 */

void DynamicStateSegmentation::computeFPFH(
    pcl::PointCloud<FPFH>::Ptr fpfhs, const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals, const float radius) const {
    if (cloud->empty() || normals->empty() ||
        cloud->size() != normals->size()) {
       ROS_ERROR("ERROR: CANNOT COMPUTE FPFH");
       return;
    }
    pcl::FPFHEstimationOMP<PointT, NormalT, FPFH> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(radius);
    fpfh.compute(*fpfhs);
}

void DynamicStateSegmentation::pointColorContrast(
    pcl::PointCloud<PointT>::Ptr region,
    const pcl::PointCloud<PointT>::Ptr cloud, const int scale) {
    if (cloud->empty() || region->empty() || scale == 0) {
        ROS_ERROR("ERROR COMPUTING COLOR CONSTRAST");
        return;
    }

    // TODO: instead of cloud use indices from region
    float color_dist[static_cast<int>(region->size())];
    for (int i = 0; i < region->size(); i++) {
        color_dist[i] = 0.0f;
    }
    float radius = 0.005f;
    float change_incf = 0.005f;
    int icount = 0;

    float max_val = 0.0f;
    do {
        for (int i = 0; i < region->size(); i++) {
           PointT pt = region->points[i];
           std::vector<int> neigbor_indices;
           this->getPointNeigbour<float>(neigbor_indices, cloud, pt,
                                         radius, false);
           
           float dist = 0.0f;
           for (int j = 1; j < neigbor_indices.size(); j++) {
              PointT n_pt = cloud->points[neigbor_indices[j]];
              dist += this->intensitySimilarityMetric<float>(pt, n_pt);
           }
           color_dist[i] += dist;

           if (color_dist[i] > max_val) {
              max_val = color_dist[i];
           }
        }
        radius += change_incf;
    } while (icount++ < scale);

    for (int i = 0; i < region->size(); i++) {
        region->points[i].r = (color_dist[i] / max_val) * 255.0f;
        region->points[i].b = (color_dist[i] / max_val) * 255.0f;
        region->points[i].g = (color_dist[i] / max_val) * 255.0f;
    }
}

template<class T>
T DynamicStateSegmentation::intensitySimilarityMetric(
    const PointT pt, const PointT n_pt, bool is_root) {
    T r = std::pow((255.0 - pt.r - n_pt.r) / 255.0, 2);
    T g = std::pow((255.0 - pt.g - n_pt.g) / 255.0, 2);
    T b = std::pow((255.0 - pt.b - n_pt.b) / 255.0, 2);
    if (!is_root) {
       return (r + g + b);
    }
    return std::sqrt(r + g + b);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "dynamic_state_segmentation");
    DynamicStateSegmentation dss;
    ros::spin();
    return 0;
}
