
#include <dynamic_state_segmentation/dynamic_state_segmentation.h>

DynamicStateSegmentation::DynamicStateSegmentation() :
    num_threads_(16), neigbor_size_(50) {
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    this->onInit();
}

void DynamicStateSegmentation::onInit() {
    this->subscribe();

    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
        "target", 1);
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

    pcl::PointCloud<PointT>::Ptr seed_region(new pcl::PointCloud<PointT>);
    for (int i = 0; i < labels.size(); i++) {
        if (labels[i] != -1) {
            PointT pt = cloud->points[i];
            seed_region->push_back(pt);
        }
    }
    
    ROS_INFO("DONE.");
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*seed_region, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    // this->pub_cloud_.publish(ros_cloud);



    /**
     * TEST
     */

    // this->pointColorContrast(seed_region, cloud, 3);
    this->normalEdge(seed_region, 3);
    
    pcl::toROSMsg(*seed_region, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
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
    float norm_similarity = (M_PI - std::acos(c_normal.dot(n_normal))) / M_PI;
    
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

void DynamicStateSegmentation::computeFeatures(
    cv::Mat &histogram, const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<NormalT>::Ptr normals, const int index) {
    if (cloud->empty() || normals->empty() || cloud->size() != normals->size()) {
        ROS_ERROR("THE CLOUD IS EMPTY. RETURING VOID IN FEATURES");
        return;
    }
    // TODO: COMPUTE FEATURES IF REQUIRED
    
}


/**
 * DESCRIPTORS
 */

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
    const PointT pt, const PointT n_pt) {
    T r = std::pow((255.0 - pt.r - n_pt.r) / 255.0, 2);
    T g = std::pow((255.0 - pt.g - n_pt.g) / 255.0, 2);
    T b = std::pow((255.0 - pt.b - n_pt.b) / 255.0, 2);
    return std::sqrt(r + g + b);
}

void DynamicStateSegmentation::normalEdge(
    pcl::PointCloud<PointT>::Ptr cloud, const float radi2) const {
    if (cloud->empty() && radi2 != 0.0f) {
	ROS_ERROR("INCORRECT DIM FOR EDGE DETECTION");
	return;
    }
    pcl::search::Search<PointT>::Ptr tree;
    pcl::NormalEstimationOMP<PointT, pcl::PointNormal> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setViewPoint(std::numeric_limits<float>::max(),
		    std::numeric_limits<float>::max(),
		    std::numeric_limits<float>::max());
    const float scale1 = 0.02f;
    ne.setRadiusSearch(scale1);
    pcl::PointCloud<pcl::PointNormal>::Ptr normals1(new pcl::PointCloud<pcl::PointNormal>);
    ne.compute(*normals1);

    const float scale2 = 0.04f;
    ne.setRadiusSearch(scale2);
    pcl::PointCloud<pcl::PointNormal>::Ptr normals2(new pcl::PointCloud<pcl::PointNormal>);
    ne.compute(*normals2);    

    pcl::DifferenceOfNormalsEstimation<PointT, pcl::PointNormal, pcl::PointNormal> don;
    don.setInputCloud(cloud);
    don.setNormalScaleLarge(normals2);
    don.setNormalScaleSmall(normals1);

    pcl::PointCloud<pcl::PointNormal>::Ptr don_cloud(new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud<PointT, pcl::PointNormal>(*cloud, *don_cloud);
    don.computeFeature(*don_cloud);


    float threshold = 0.15f;
    
    pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond (
	new pcl::ConditionOr<pcl::PointNormal> ());
    range_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(
				  new pcl::FieldComparison<pcl::PointNormal>(
				      "curvature", pcl::ComparisonOps::GT, threshold)));
    pcl::ConditionalRemoval<pcl::PointNormal> condrem(range_cond);
    condrem.setInputCloud(don_cloud);
    pcl::PointCloud<pcl::PointNormal>::Ptr doncloud_filtered(new pcl::PointCloud<pcl::PointNormal>);
    condrem.filter(*doncloud_filtered);
    don_cloud->clear();
    don_cloud = doncloud_filtered;
    
    std::cout << "Size: " << don_cloud->size()  << "\n";
    cloud->clear();
    for (int i = 0; i < don_cloud->size(); i++) {
	PointT pt;
	pt.x = don_cloud->points[i].x;
	pt.y = don_cloud->points[i].y;	
	pt.z = don_cloud->points[i].z;
	pt.r = 255;
	cloud->push_back(pt);
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "dynamic_state_segmentation");
    DynamicStateSegmentation dss;
    ros::spin();
    return 0;
}
