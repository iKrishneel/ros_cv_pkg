
#include <dynamic_state_segmentation/dynamic_state_segmentation.h>

DynamicStateSegmentation::DynamicStateSegmentation() :
    num_threads_(16), neigbor_size_(50) {
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    this->onInit();
}

void DynamicStateSegmentation::onInit() {
    this->srv_client_ = this->pnh_.serviceClient<
	dynamic_state_segmentation::Feature3DClustering>("feature3d_clustering_srv");
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

    // BUILD FEATURE VECTOR HERE
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
    

    // pcl::toROSMsg(*seed_cloud, ros_cloud);
    // ros_cloud.header = cloud_msg->header;
    // this->pub_cloud_.publish(ros_cloud);

    /**
     * TEST
     */

    // this->pointColorContrast(seed_cloud, cloud, 3);
    // this->computeFeatures(seed_cloud, seed_normals, 1);

    /*
    cloud->empty();
    pcl::copyPointCloud<PointT, PointT>(*seed_cloud, *cloud);
    this->kdtree_->setInputCloud(seed_cloud);
    std::vector<pcl::PointIndices> all_indices;
    this->clusterFeatures(all_indices, seed_cloud, seed_normals, 20, 0.05);
    */

    pcl::PointCloud<PointT>::Ptr appearance_weights(new pcl::PointCloud<PointT>);
    this->appearanceKernel(appearance_weights, seed_cloud, seed_normals);
    seed_cloud->clear();
    *seed_cloud = *appearance_weights;
    
    jsk_recognition_msgs::ClusterPointIndices ros_indices;
    ros_indices.cluster_indices = pcl_conversions::convertToROSPointIndices(
	all_indices, cloud_msg->header);
    ros_indices.header = cloud_msg->header;
    this->pub_indices_.publish(ros_indices);

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*seed_cloud, ros_cloud);
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
#pragma omp parallel for num_threads(this->num_threads_) shared(merge_list, labels)
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
            if (this->localVoxelConvexityCriteria(
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

int DynamicStateSegmentation::localVoxelConvexityCriteria(
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
    float norm_similarity = (M_PI - std::acos(c_normal.dot(n_normal) /
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
    const pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr normals,
    const T k, bool use_knn) const {
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


void regionOverSegmentation(pcl::PointCloud<PointT>::Ptr region,
			    const pcl::PointCloud<PointT>::Ptr cloud) {
    if (cloud->empty() || region->empty()) {
	ROS_ERROR("EMPTY CLOUD FOR REGION OVERSEGMENTATION");
	return;
    }
    Eigen::Vector4f center;
    pcl::compute3DCentroid(region, center);
    
}

/**
 * smoothness term
 */
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

void DynamicStateSegmentation::appearanceKernel(
    pcl::PointCloud<PointT>::Ptr weights,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<NormalT>::Ptr normals) {
    if (cloud->empty() || cloud->size() != normals->size()) {
	ROS_ERROR("INAPPOPRIATE SIZE");
	return;
    }
    pcl::copyPointCloud<PointT, PointT>(*cloud, *weights);
    const float weight_param = 1.0f;
    const float position_param = 1.0f;
    const float color_param = 1.0f;
    for (int i = 0; i < cloud->size(); i++) {
	std::vector<int> neigbor_indices;
	this->getPointNeigbour<int>(neigbor_indices, cloud, cloud->points[i], 8);
	float sum = 0.0f;
	Eigen::Vector3f center_pt = cloud->points[i].getVector3fMap();
	for (int j = 0; j < neigbor_indices.size(); j++) {

	    float convex_term = (cloud->points[j].getVector3fMap() - center_pt).dot(
		normals->points[j].getNormalVector3fMap());
	    float norm_dif = normals->points[i].getNormalVector4fMap().dot(
		normals->points[j].getNormalVector4fMap()) / (
		    normals->points[i].getNormalVector4fMap().norm() *
		    normals->points[j].getNormalVector4fMap().norm());
	    float v_term = 0.0f;
	    if (convex_term > -0.02) {
		v_term = std::exp(-norm_dif/(2 * M_PI));
	    } else {
		v_term = std::exp(-norm_dif/(M_PI/3));
	    }

	    float curvature1 = normals->points[i].curvature;
	    float curvature2 = normals->points[j].curvature;
	    
	    // sum += v_term;
	    // sum += (std::exp(-1 * (curvature1/curvature2)));
	    float curvature = (((curvature1 / curvature2) * (curvature1 / curvature2)) / (2.0f * 1.0));
	    sum += (1 * std::exp(- curvature) + (1 * v_term));



	    
	    float pos_dif = this->distancel2<float>(cloud->points[j].getVector3fMap(),
						    center_pt, true);
	    float position = (pos_dif * pos_dif) / (2.0f * position_param * position_param);

	    float col_dif = this->intensitySimilarityMetric<float>(cloud->points[i],
	       							   cloud->points[j], true);
	    
	    norm_dif = std::acos(norm_dif);
	    // float col_dif = norm_dif;
	    
	    float intensity = (col_dif * col_dif) / (2.0f * color_param * color_param);
	    float app_kernel = std::exp(-position - intensity) * 0.5f;
	    
	    // smoothness
	    float smoothness = weight_param * std::exp(-position);
	    
	    // sum += (app_kernel + smoothness);
	}

	// std::cout << sum  << "\t" << static_cast<float>(neigbor_indices.size())  << "\n";
	float avg = sum / static_cast<float>(neigbor_indices.size());
	// float avg = sum;
	
	// std::cout << avg  << "\t" << static_cast<float>(neigbor_indices.size())  << "\n";
	
	weights->points[i].r = (avg * 255);
	weights->points[i].g = (avg * 255);	
	weights->points[i].b = (avg * 255);
    }
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
    std::vector<std::vector<int> > neigbor_cache(static_cast<int>(descriptors->size()));
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
	// hist.histogram.push_back(coplanar/static_cast<float>(neigbor_indices.size()));
	hist.histogram.push_back(curvature/static_cast<float>(neigbor_indices.size()));
	
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
    pcl::PointCloud<NormalT>::Ptr normals, const std::vector<std::vector<int> > neigbor_cache) {
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
    pcl::PointCloud<PointT>::Ptr cloud, const pcl::PointCloud<NormalT>::Ptr normals,
    const int index1) {
    if (cloud->empty() || normals->empty() || cloud->size() != normals->size()) {
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
    
    do{
	for (int i = 0; i < cloud->size(); i++) {
	    std::vector<int> neigbor_indices;

	    this->getPointNeigbour<float>(neigbor_indices, cloud, cloud->points[i],
	     				  search_radius, false);
	    // this->getPointNeigbour<int>(neigbor_indices, cloud, cloud->points[i],
	    // 				  search_k, true);
	    
	
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
	this->getPointNeigbour<int>(cache_neigbour[i], cloud, cloud->points[i], knn);
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
    /*
    T sum = 0.0;
    for (int i = 0; i < bin_size; i++) {
	float a = isnan(histA.histogram[i]) ? 0.0 : histA.histogram[i];
	float b = isnan(histB.histogram[i]) ? 0.0 : histB.histogram[i];
	sum += (a + b - std::abs(a - b));
	// sum += (std::min(a, b));
    }
    return (0.5 * sum);
    */
    
    FPFH ha = histA;
    FPFH hb = histB;
    cv::Mat a = cv::Mat(1, 33, CV_32FC1, ha.histogram);
    cv::Mat b = cv::Mat(1, 33, CV_32FC1, hb.histogram);
    T dist = static_cast<T>(cv::compareHist(a, b, CV_COMP_INTERSECT));
    
    return (dist);
    
}

/**
 * DESCRIPTORS
 */

void DynamicStateSegmentation::computeFPFH(
    pcl::PointCloud<FPFH>::Ptr fpfhs, const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals, const float radius) const {
    if (cloud->empty() || normals->empty() || cloud->size() != normals->size()) {
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
    } while(icount++ < scale);

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
