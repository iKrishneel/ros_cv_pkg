// Copyright (C) 2015 by Krishneel Chaudhary, JSK Lab,
// The University of Tokyo, Japan

#include <interactive_segmentation/interactive_segmentation.h>
#include <vector>

InteractiveSegmentation::InteractiveSegmentation():
    min_cluster_size_(100), is_init_(true),
    num_threads_(8) {
    pnh_.getParam("num_threads", this->num_threads_);

    // this->srv_client_ = this->pnh_.serviceClient<
    //   interactive_segmentation::OutlierFiltering>("outlier_filtering_srv");
    
    this->subscribe();
    this->onInit();
}

void InteractiveSegmentation::onInit() {
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/interactive_segmentation/output/cloud", 1);

    this->pub_prob_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/interactive_segmentation/output/selected_probability", 1);
    
    this->pub_indices_ = this->pnh_.advertise<
       jsk_recognition_msgs::ClusterPointIndices>(
          "/interactive_segmentation/output/indices", 1);
    
    this->pub_voxels_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
          "/interactive_segmentation/output/anchor_points", 1);

    this->pub_normal_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/interactive_segmentation/output/normal", 1);
    
    this->pub_pt_map_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
        "/interactive_segmentation/output/point_map", 1);

    this->pub_concave_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
        "/interactive_segmentation/output/concave_edge", 1);

    this->pub_convex_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
        "/interactive_segmentation/output/convex_edge", 1);
    
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/interactive_segmentation/output/image", 1);
}

void InteractiveSegmentation::subscribe() {

       this->sub_screen_pt_.subscribe(this->pnh_, "input_screen", 1);
       this->sub_orig_cloud_.subscribe(this->pnh_, "input_orig_cloud", 1);
       this->usr_sync_ = boost::make_shared<message_filters::Synchronizer<
         UsrSyncPolicy> >(100);
       usr_sync_->connectInput(sub_screen_pt_, sub_orig_cloud_);
       usr_sync_->registerCallback(boost::bind(
           &InteractiveSegmentation::screenPointCallback, this, _1, _2));
       
       this->sub_image_.subscribe(this->pnh_, "input_image", 1);
       this->sub_info_.subscribe(this->pnh_, "input_info", 1);
       this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
       this->sub_normal_.subscribe(this->pnh_, "input_normal", 1);
       this->sync_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy> >(100);
       sync_->connectInput(sub_image_, sub_info_, sub_cloud_, sub_normal_);
       sync_->registerCallback(boost::bind(&InteractiveSegmentation::callback,
                                           this, _1, _2, _3, _4));
}

void InteractiveSegmentation::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_info_.unsubscribe();
    this->sub_image_.unsubscribe();
}

void InteractiveSegmentation::screenPointCallback(
    const geometry_msgs::PointStamped::ConstPtr &screen_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    int x = screen_msg->point.x;
    int y = screen_msg->point.y;
    this->screen_pt_ = cv::Point2i(x, y);
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->height == 1) {
      return;
    }
    const int index = x + (y * cloud->width);
    this->user_marked_pt_ = cloud->points[index];
    if (!isnan(user_marked_pt_.x) &&
        !isnan(user_marked_pt_.y) &&
        !isnan(user_marked_pt_.z)) {
      this->is_init_ = true;
    }
}

void InteractiveSegmentation::callback(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const sensor_msgs::CameraInfo::ConstPtr &info_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::PointCloud2::ConstPtr &orig_cloud_msg) {
    boost::mutex::scoped_lock lock(this->mutex_);
    cv::Mat image = cv_bridge::toCvShare(
       image_msg, image_msg->encoding)->image;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    pcl::PointCloud<PointT>::Ptr original_cloud(
       new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*orig_cloud_msg, *original_cloud);

    this->camera_info_ = info_msg;
    
    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud<PointT>(*cloud, *cloud, nan_indices);
        
    ROS_INFO("\033[32m DEBUG: PROCESSING CALLBACK \033[0m");
    
    // ----------------------------------------
    int k = 100;  // thresholds
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    this->estimatePointCloudNormals<int>(cloud, normals, k, true);
    
    pcl::PointCloud<PointT>::Ptr concave_edge_points(
       new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr convex_edge_points(
       new pcl::PointCloud<PointT>);
    this->highCurvatureEdgeBoundary(concave_edge_points, convex_edge_points,
                                    cloud, normals, cloud_msg->header);
    
    pcl::PointCloud<PointT>::Ptr anchor_points(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(*cloud, *anchor_points);
    pcl::PointIndices::Ptr anchor_indices(new pcl::PointIndices);
    bool is_found_points = this->estimateAnchorPoints(
       anchor_points, convex_edge_points, concave_edge_points,
       anchor_indices, original_cloud, cloud_msg->header);
    
    this->publishAsROSMsg(anchor_points, pub_voxels_, cloud_msg->header);
    this->publishAsROSMsg(concave_edge_points, pub_concave_, cloud_msg->header);
    this->publishAsROSMsg(convex_edge_points, pub_convex_, cloud_msg->header);
    
    // ---------------------PROCESSING-------------------
    ROS_INFO("\033[32m LABELING ON THE POINT \033[0m");

    if (is_found_points) {
       this->selectedVoxelObjectHypothesis(
          cloud, normals, anchor_indices, cloud_msg->header);
    }
    
    // ----------------------END-PROCESSING------------------

    ROS_INFO("\n\033[34m ALL VALID REGION LABELED \033[0m");
}

void InteractiveSegmentation::selectedVoxelObjectHypothesis(
    const pcl::PointCloud<PointT>::Ptr in_cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    const pcl::PointIndices::Ptr indices,
    const std_msgs::Header header) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    const int mem_size = indices->indices.size();
    // std::vector<pcl::PointCloud<PointT>::Ptr> anchor_points_weights(mem_size);

    std::vector<cv::Mat> anchor_points_weights(mem_size);
    std::vector<float> anchor_points_max(mem_size);
    
    for (int i = 0; i < indices->indices.size(); i++) {
       cloud->clear();
       pcl::copyPointCloud<PointT, PointT>(*in_cloud, *cloud);
       int index = indices->indices[i];
       Eigen::Vector4f attention_normal = normals->points[
          index].getNormalVector4fMap();
       Eigen::Vector4f attention_centroid = cloud->points[
          index].getVector4fMap();

       std::cout << "\033[34m COMPUTING WEIGHTS \033[0m" << std::endl;

       cv::Mat weight_map;
       this->surfelSamplePointWeightMap(cloud, normals, cloud->points[index],
                                        attention_normal, weight_map);

       pcl::PointCloud<PointT>::Ptr weight_cloud(
          new pcl::PointCloud<PointT>);
       float max_weight = 0.0f;
       for (int x = 0; x < weight_map.rows; x++) {
          if (weight_map.at<float>(x, 0) > max_weight) {
             max_weight = weight_map.at<float>(x, 0);
          }
          cloud->points[x].r = weight_map.at<float>(x, 0) * 255.0f;
          cloud->points[x].g = weight_map.at<float>(x, 0) * 255.0f;
          cloud->points[x].b = weight_map.at<float>(x, 0) * 255.0f;
       }
       *weight_cloud = *cloud;
       
       std::cout << "\033[34m OBJECT MASK EXTRACTION \033[0m"
                 << std::endl;

       /*
       pcl::PointCloud<PointT>::Ptr prob_object_cloud(
          new pcl::PointCloud<PointT>);
       pcl::PointIndices::Ptr prob_object_indices(new pcl::PointIndices);
       this->attentionSurfelRegionPointCloudMask(cloud, attention_centroid,
                                                 header, prob_object_cloud,
                                                 prob_object_indices);
       */
                                                 
       // anchor_points_weights[i] = pcl::PointCloud<PointT>::Ptr(
       //    new pcl::PointCloud<PointT>);
       // pcl::copyPointCloud<PointT, PointT>(*cloud, *anchor_points_weights[i]);

       anchor_points_weights[i] = weight_map;
       anchor_points_max[i] = max_weight;
       
       publishAsROSMsg(cloud, pub_cloud_, header);
       // ros::Duration(5).sleep();
    }
    
    // TODO(HERE): combine the weight maps
    cv::Mat conv_weight_map = cv::Mat::zeros(
       static_cast<int>(in_cloud->size()), 1, CV_32F);
    for (int i = 0; i < anchor_points_weights.size(); i++) {
       cv::Scalar mean;
       cv::Scalar stddev;
       cv::meanStdDev(anchor_points_weights[i], mean, stddev);
       float psr = (anchor_points_max[i] - mean.val[0])/stddev.val[0];
       cv::Mat weight_map = anchor_points_weights[i];
       for (int j = 0; j < weight_map.rows; j++) {
          conv_weight_map.at<float>(j, 0) += (weight_map.at<float>(j, 0) * psr);
       }
       anchor_points_weights[i] = weight_map;
       // conv_weight_map *= weight_map;
    }
    cv::normalize(conv_weight_map, conv_weight_map, 0, 1,
                  cv::NORM_MINMAX, -1, cv::Mat());
    
    pcl::PointCloud<PointT>::Ptr weight_cloud(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(*in_cloud, *weight_cloud);
    for (int j = 0; j < conv_weight_map.rows; j++) {
       float w = (conv_weight_map.at<float>(j, 0) / 1.0f) *  255.0f;
       weight_cloud->points[j].r = w;
       weight_cloud->points[j].g = w;
       weight_cloud->points[j].b = w;
    }
    publishAsROSMsg(weight_cloud, pub_cloud_, header);
    // ros::Duration(10).sleep();
}


void InteractiveSegmentation::surfelSamplePointWeightMap(
     const pcl::PointCloud<PointT>::Ptr cloud,
     const pcl::PointCloud<pcl::Normal>::Ptr normals, const PointT &centroid_pt,
     const Eigen::Vector4f attention_normal, cv::Mat &weights
     /*pcl::PointCloud<PointT>::Ptr weights*/) {
     if (cloud->empty() || normals->empty()) {
       return;
     }
     Eigen::Vector4f attention_centroid = centroid_pt.getVector4fMap();
     cv::Mat connectivity_weights = cv::Mat::zeros(normals->size(), 1, CV_32F);
     // cv::Mat orientation_weights;
     cv::Mat orientation_weights = cv::Mat::zeros(normals->size(), 1, CV_32F);

#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_)
#endif
     for (int i = 0; i < normals->size(); i++) {
       Eigen::Vector4f current_pt = cloud->points[i].getVector4fMap();
       Eigen::Vector4f d = (attention_centroid - current_pt) /
           (attention_centroid - current_pt).norm();
       Eigen::Vector4f current_normal =
           normals->points[i].getNormalVector4fMap();
       // float connection = (attention_normal - current_normal).dot(d);
       float connection = (current_pt - attention_centroid).dot(
           current_normal);
       if (connection < 0.0f || isnan(connection)) {
          connection = 0.0f;
       } else {
         // connection = acos(current_normal.dot(attention_normal))/
         //     (1.0f * M_PI);
          connection = std::pow((current_normal.dot(attention_normal)), 2);
       }
       // connectivity_weights.push_back(connection);
       connectivity_weights.at<float>(i, 0) = connection;
       /*     
       Eigen::Vector3f view_point_vec = (cloud->points[i].getVector3fMap() -
                                         centroid_pt.getVector3fMap());
       */
       
       Eigen::Vector3f view_point_vec = (cloud->points[i].getVector3fMap() -
                                         centroid_pt.getVector3fMap());
       Eigen::Vector3f surface_normal_vec = normals->points[
          i].getNormalVector3fMap();
       /*
       Eigen::Vector3f surface_normal_vec = normals->points[
          i].getNormalVector3fMap() - attention_normal.head<3>();
       */


       // TODO(HERE):  add Gaussian centered at selected
       float cross_norm = static_cast<float>(
           surface_normal_vec.cross(view_point_vec).norm());
       float scalar_prod = static_cast<float>(
           surface_normal_vec.dot(view_point_vec));
       float angle = atan2(cross_norm, scalar_prod);
       
       float view_pt_weight = (CV_PI - angle)/(1.0 * CV_PI);
       // view_pt_weight = 1.0f / (1.0f + (view_pt_weight * view_pt_weight));
       // view_pt_weight = std::exp(-1.0f * view_pt_weight);
       if (isnan(angle)) {
          view_pt_weight = 0.0f;
       }

       // TODO(HERE): CENTERED GAUSSIAN WEIGHTS
       
       // view_pt_weight *= this->whiteNoiseKernel(view_pt_weight);
       view_pt_weight *= this->whiteNoiseKernel(view_pt_weight, 0.0f, 15.0f);
       // orientation_weights.push_back(view_pt_weight);
       orientation_weights.at<float>(i, 0) = view_pt_weight;
     }
     /*
     cv::normalize(connectivity_weights, connectivity_weights, 0, 1,
                   cv::NORM_MINMAX, -1, cv::Mat());
     cv::normalize(orientation_weights, orientation_weights, 0, 1,
                   cv::NORM_MINMAX, -1, cv::Mat());
                   
     // smoothing HERE
     /*
     const int filter_lenght = 5;
     cv::GaussianBlur(connectivity_weights, connectivity_weights,
                      cv::Size(filter_lenght, filter_lenght), 0, 0);
     cv::GaussianBlur(orientation_weights, orientation_weights,
                      cv::Size(filter_lenght, filter_lenght), 0.0,
     0.0);
     */
     /*
     // morphological
     int erosion_size = 5;
     cv::Mat element = cv::getStructuringElement(
         cv::MORPH_ELLIPSE,
         cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
         cv::Point(erosion_size, erosion_size));
     cv::dilate(connectivity_weights, connectivity_weights, element);
     cv::dilate(orientation_weights, orientation_weights, element);
     cv::erode(connectivity_weights, connectivity_weights, element);
     cv::erode(orientation_weights, orientation_weights, element);
     */
        
     // convolution of distribution
     // pcl::copyPointCloud<PointT, PointT>(*cloud, *weights);
     weights = cv::Mat::zeros(static_cast<int>(cloud->size()), 1, CV_32F);
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_) shared(weights)
#endif
     for (int i = 0; i < connectivity_weights.rows; i++) {
       float pix_val = connectivity_weights.at<float>(i, 0);
       connectivity_weights.at<float>(i, 0) = pix_val *
           this->whiteNoiseKernel(pix_val);
       // pix_val *= this->whiteNoiseKernel(pix_val);
       pix_val *= orientation_weights.at<float>(i, 0);

       // pix_val = orientation_weights.at<float>(i, 0);
       
       if (isnan(pix_val)) {
          pix_val = 0.0f;
       }
       weights.at<float>(i, 0) = pix_val;
       
       // weights->points[i].r = pix_val * 255.0f;
       // weights->points[i].b = pix_val * 255.0f;
       // weights->points[i].g = pix_val * 255.0f;
     }
}


bool InteractiveSegmentation::attentionSurfelRegionPointCloudMask(
    const pcl::PointCloud<PointT>::Ptr weight_cloud,
    const Eigen::Vector4f centroid, const std_msgs::Header header,
    pcl::PointCloud<PointT>::Ptr prob_object_cloud,
    pcl::PointIndices::Ptr prob_object_indices) {
    if (weight_cloud->empty()) {
      return false;
    }
    // removed zero points
    pcl::PointIndices::Ptr prob_indices(new pcl::PointIndices);
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    const float threshold = 0.0f * 255.0f;
    for (int i = 0; i < weight_cloud->size(); i++) {
       PointT pt = weight_cloud->points[i];
       if (pt.r > threshold && pt.b > threshold && pt.g > threshold &&
           !isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z)) {
          cloud->push_back(pt);
          prob_indices->indices.push_back(i);
       }
    }
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    cluster_indices.clear();
    pcl::EuclideanClusterExtraction<PointT> euclidean_clustering;
    euclidean_clustering.setClusterTolerance(0.01f);
    euclidean_clustering.setMinClusterSize(this->min_cluster_size_);
    euclidean_clustering.setMaxClusterSize(25000);
    euclidean_clustering.setSearchMethod(tree);
    euclidean_clustering.setInputCloud(weight_cloud);
    euclidean_clustering.setIndices(prob_indices);
    euclidean_clustering.extract(cluster_indices);
    double min_distance = DBL_MAX;
    pcl::ExtractIndices<PointT>::Ptr eifilter(new pcl::ExtractIndices<PointT>);
    eifilter->setInputCloud(weight_cloud);
    for (int i = 0; i < cluster_indices.size(); i++) {
       pcl::PointIndices::Ptr region_indices(new pcl::PointIndices);
       *region_indices = cluster_indices[i];
       eifilter->setIndices(region_indices);
       pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
       eifilter->filter(*tmp_cloud);
       Eigen::Vector4f center;
       pcl::compute3DCentroid<PointT, float>(*tmp_cloud, center);
       double dist = pcl::distances::l2(centroid, center);
       if (dist < min_distance) {
          min_distance = dist;
          prob_object_indices->indices.clear();
          *prob_object_indices = *region_indices;
          prob_object_cloud->clear();
          *prob_object_cloud = *tmp_cloud;
       }
    }
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*prob_object_cloud, ros_cloud);
    ros_cloud.header = header;
    this->pub_prob_.publish(ros_cloud);
}

void InteractiveSegmentation::generateFeatureSaliencyMap(
    const cv::Mat &img, cv::Mat &saliency_img) {
    cv::Mat image = img.clone();
    cv::cvtColor(img, image, CV_BGR2GRAY);
    SaliencyMapGenerator saliency_map(8);
    saliency_map.computeSaliencyImpl(image, saliency_img);
    cv::cvtColor(saliency_img, saliency_img, CV_GRAY2BGR);
}

void InteractiveSegmentation::surfelLevelObjectHypothesis(
    pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<pcl::Normal>::Ptr normals,
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > &convex_supervoxels) {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: EMPTY CLOUD");
       return;
    }
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
    AdjacencyList adjacency_list;
    this->supervoxelSegmentation(cloud,
                                 supervoxel_clusters,
                                 adjacency_list);
    std::map<uint32_t, int> voxel_labels;
    convex_supervoxels.clear();
    // cloud->clear();
    for (std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>::iterator it =
            supervoxel_clusters.begin(); it != supervoxel_clusters.end();
         it++) {
       voxel_labels[it->first] = -1;
       pcl::Supervoxel<PointT>::Ptr supervoxel =
          supervoxel_clusters.at(it->first);
       // *normals = *normals + *(supervoxel->normals_);
       // *cloud = *cloud + *(supervoxel->voxels_);
    }
    int label = -1;
    AdjacencyList::vertex_iterator i, end;
    for (boost::tie(i, end) = boost::vertices(adjacency_list); i != end; i++) {
       AdjacencyList::adjacency_iterator ai, a_end;
       boost::tie(ai, a_end) = boost::adjacent_vertices(*i, adjacency_list);
       uint32_t vindex = static_cast<int>(adjacency_list[*i]);
       
       Eigen::Vector4f v_normal = this->cloudMeanNormal(
          supervoxel_clusters.at(vindex)->normals_);
       // Eigen::Vector4f v_normal = supervoxel_clusters.at(
       //     vindex)->normal_.getNormalVector4fMap();
       std::map<uint32_t, int>::iterator it = voxel_labels.find(vindex);
       if (it->second == -1) {
          voxel_labels[vindex] = ++label;
       }
       std::vector<uint32_t> neigb_ind;
       for (; ai != a_end; ai++) {
          bool found = false;
          AdjacencyList::edge_descriptor e_descriptor;
          boost::tie(e_descriptor, found) = boost::edge(
             *i, *ai, adjacency_list);
          if (found) {
             float weight = adjacency_list[e_descriptor];
             uint32_t n_vindex = adjacency_list[*ai];
             float conv_criteria = (
                supervoxel_clusters.at(vindex)->centroid_.getVector4fMap() -
                supervoxel_clusters.at(n_vindex)->centroid_.getVector4fMap()).
                dot(v_normal);
             neigb_ind.push_back(n_vindex);
             if (conv_criteria <= this->convex_threshold_ ||
                 isnan(conv_criteria)) {
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
                   }
                   boost::remove_edge(n_edge, adjacency_list);
                }
                boost::clear_vertex(*ai, adjacency_list);
                voxel_labels[n_vindex] = label;
             }
          }
       }
       convex_supervoxels[vindex] = supervoxel_clusters.at(vindex);
    }
    supervoxel_clusters.clear();
}


void InteractiveSegmentation::computePointCloudCovarianceMatrix(
    const pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<PointT>::Ptr new_cloud) {
    if (cloud->empty()) {
       return;
    }
    float knearest = 0.03f;
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);

    std::vector<float> eigen_entropy(cloud->size());
    float *eigen_entropy_ptr = &eigen_entropy[0];
    std::vector<float> sum_of_eigens(cloud->size());
    float *sum_of_eigens_ptr = &sum_of_eigens[0];
    std::vector<float> curvature(cloud->size());
    float *curvature_ptr = &curvature[0];

    std::vector<Eigen::Vector4f> principle_axis;
    
    //******************
    // TODO(HERE): how to keep the attention_pt
    Eigen::Vector4f center;
    pcl::compute3DCentroid<PointT, float>(*cloud, center);
    double distance = DBL_MAX;
    int index = -1;
    
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(this->num_threads_) \
//     shared(kdtree, eigen_entropy_ptr, sum_of_eigens_ptr, curvature_ptr)
// #endif
    for (int i = 0; i < cloud->size(); i++) {
       PointT centroid_pt = cloud->points[i];
       if (!isnan(centroid_pt.x) || !isnan(centroid_pt.y) ||
           !isnan(centroid_pt.z)) {
          std::vector<int> point_idx_search;
          std::vector<float> point_squared_distance;
          int search_out = kdtree.radiusSearch(
          centroid_pt, knearest, point_idx_search, point_squared_distance);
          Eigen::Matrix3f covariance_matrix;
          uint32_t num_pt = pcl::computeCovarianceMatrix<PointT, float>(
             *cloud, point_idx_search,
             centroid_pt.getVector4fMap(),
             covariance_matrix);

          Eigen::Vector3f eigen_vals;  // 2 > 1 > 0
          Eigen::Matrix3f eigen_vects;
          pcl::eigen33(covariance_matrix, eigen_vects, eigen_vals);


          // std::cout << eigen_vects(0, 2) << std::endl;
          // std::cout << eigen_vects << "\n\n";
          Eigen::Vector4f vec = Eigen::Vector4f(eigen_vects(0, 2),
                                                eigen_vects(1, 2),
                                                eigen_vects(2, 2), 1.0f);
          principle_axis.push_back(vec);
          
          
          // eigen entropy
          float sum_entropy = 0.0f;
          float sum_eigen = 0.0f;
          for (int j = 0; j < 3; j++) {
             sum_entropy += eigen_vals(j) * std::log(eigen_vals(j));
             sum_eigen += eigen_vals(j);
          }
          eigen_entropy_ptr[i] = sum_entropy * -1.0f;
          sum_of_eigens_ptr[i] = sum_eigen;
          // sum_of_eigens_ptr[i] = (eigen_vals(0) - eigen_vals(2)) /
          // eigen_vals(1);
          
          
          curvature_ptr[i] = (eigen_vals(0) / sum_eigen);

          // ****** TEMP
          double dist = pcl::distances::l2(
             centroid_pt.getVector4fMap(), center);
          if (dist < distance) {
             index = i;
          }
       }
    }

    
    // pcl::PointCloud<PointT>::Ptr new_cloud(new
    // pcl::PointCloud<PointT>);
    for (int i = 0; i < cloud->size(); i++) {
      float dist_diff_sum = std::pow(
         sum_of_eigens_ptr[index] - sum_of_eigens_ptr[i], 2);
      dist_diff_sum += dist_diff_sum;
      dist_diff_sum = std::sqrt(dist_diff_sum);
      // std::cout << "DIST: " << dist_diff_sum  << "\t";
       
      float dist_diff_cur = std::pow(
         curvature_ptr[index] - curvature_ptr[i], 2);
      dist_diff_cur += dist_diff_cur;
      dist_diff_cur = std::sqrt(dist_diff_cur);
      
       float dist_diff_ent = eigen_entropy_ptr[index] / eigen_entropy_ptr[i];

       dist_diff_sum += (dist_diff_cur + dist_diff_ent);
       // dist_diff_sum = (dist_diff_cur);
       // std::cout << "DIST: " << dist_diff_sum  << "\n";



       double angle = pcl::getAngle3D(principle_axis[index], principle_axis[i]);
       dist_diff_sum = angle;
       float prob_sum = 1.0f/(1.0f + 0.5 * (dist_diff_sum * dist_diff_sum));


       // float prob_cur = 1.0f/(1.0f + (dist_diff_cur * dist_diff_cur));
       // float prob_ent = 1.0f/(1.0f + (dist_diff_ent * dist_diff_ent));
       // if (angle < (M_PI / 2)) {
       //   prob_sum = 0.0f;
       // }
       
       float prob = prob_sum;
       PointT pt = cloud->points[i];
       pt.r = prob * pt.r;
       pt.g = prob * pt.g;
       pt.b = prob * pt.b;
       new_cloud->push_back(pt);
    }
}




void InteractiveSegmentation::updateSupervoxelClusters(
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> &supervoxel_clusters,
    const uint32_t vindex, const uint32_t n_vindex) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    *cloud = *(supervoxel_clusters.at(vindex)->voxels_) +
       *(supervoxel_clusters.at(n_vindex)->voxels_);
    pcl::PointCloud<pcl::Normal>::Ptr normals(
       new pcl::PointCloud<pcl::Normal>);
    *normals = *(supervoxel_clusters.at(vindex)->normals_) +
       *(supervoxel_clusters.at(n_vindex)->normals_);
    // Eigen::Vector4f centre;
    // pcl::compute3DCentroid<PointT, float>(*cloud, centre);
    // pcl::PointXYZRGBA centroid;
    // centroid.x = centre(0);
    // centroid.y = centre(1);
    // centroid.z = centre(2);
    pcl::PointXYZRGBA centroid;
    pcl::PointXYZRGBA vcent = supervoxel_clusters.at(vindex)->centroid_;
    pcl::PointXYZRGBA n_vcent = supervoxel_clusters.at(n_vindex)->centroid_;
    centroid.x = (vcent.x - n_vcent.x)/2 + n_vcent.x;
    centroid.y = (vcent.y - n_vcent.y)/2 + n_vcent.y;
    centroid.z = (vcent.z - n_vcent.z)/2 + n_vcent.z;
    centroid.r = (vcent.r - n_vcent.r)/2 + n_vcent.r;
    centroid.g = (vcent.g - n_vcent.g)/2 + n_vcent.g;
    centroid.b = (vcent.b - n_vcent.b)/2 + n_vcent.b;
    centroid.a = (vcent.a - n_vcent.a)/2 + n_vcent.a;
    *(supervoxel_clusters.at(vindex)->voxels_) = *cloud;
    *(supervoxel_clusters.at(vindex)->normals_) = *normals;
    supervoxel_clusters.at(vindex)->centroid_ = centroid;
    *(supervoxel_clusters.at(n_vindex)->voxels_) = *cloud;
    *(supervoxel_clusters.at(n_vindex)->normals_) = *normals;
    supervoxel_clusters.at(n_vindex)->centroid_ = centroid;
}


void InteractiveSegmentation::pointIntensitySimilarity(
    pcl::PointCloud<PointT>::Ptr cloud,
    const int index) {
    if (cloud->empty()) {
       return;
    }
    PointT pt = cloud->points[index];
    pcl::PointXYZHSV attention_hsv;
    pcl::PointXYZRGBtoXYZHSV(cloud->points[index], attention_hsv);
    const float h_norm = 180.f;
    const float s_norm = 255.0f;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < cloud->size(); i++) {
       pcl::PointXYZHSV hsv;
       pcl::PointXYZRGBtoXYZHSV(cloud->points[i], hsv);
       float dist_color = std::sqrt(
          std::pow((hsv.h/h_norm - attention_hsv.h/h_norm), 2) +
          std::pow((hsv.s/s_norm - attention_hsv.s/s_norm), 2));
       float pix_val = exp(-1.0f * dist_color);
       // pix_val *= this->whiteNoiseKernel(pix_val);
       
       cloud->points[i].r = pix_val * 255.0f;
       cloud->points[i].g = pix_val * 255.0f;
       cloud->points[i].b = pix_val * 255.0f;
    }
}


int InteractiveSegmentation::localVoxelConvexityCriteria(
    Eigen::Vector4f c_centroid, Eigen::Vector4f n_centroid,
    Eigen::Vector4f n_normal, const float threshold) {
    if ((n_centroid - c_centroid).dot(n_normal) > 0) {
        return 1;
    } else {
        return -1;
    }
}

Eigen::Vector4f InteractiveSegmentation::cloudMeanNormal(
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
        1.0f);
    if (isnorm) {
        n_mean.normalize();
    }
    return n_mean;
}

template<class T>
void InteractiveSegmentation::estimatePointCloudNormals(
    const pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<pcl::Normal>::Ptr normals,
    const T k, bool use_knn) const {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: The Input cloud is Empty.....");
       return;
    }
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
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


float InteractiveSegmentation::whiteNoiseKernel(
    const float z, const float N, const float sigma) {

    float val = static_cast<float>(z - N);
    return static_cast<double>((1.0/(sqrt(2.0 * M_PI)) * sigma) *
                               exp(-((val * val) / (2*sigma*sigma))));

}

void InteractiveSegmentation::selectedPointToRegionDistanceWeight(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const Eigen::Vector3f attention_pts, const float step,
    const sensor_msgs::CameraInfo::ConstPtr info) {
    if (cloud->empty()) {
      return;
    }
    cv::Mat mask;
    cv::Mat depth_map;
    cv::Mat image = this->projectPointCloudToImagePlane(
       cloud, info, mask, depth_map);

    const int threshold = 5;
    cv::Point2f attent_pt = cv::Point2f(attention_pts(0), attention_pts(1));

    std::cout << "RUNNING" << std::endl;
    for (int j = 0; j < mask.rows; j++) {
       for (int i = 0; i < mask.cols; i++) {
          if (mask.at<float>(j, i) == 255) {
             float angle = std::atan2(j - attent_pt.y, i - attent_pt.x);
             float dist = cv::norm(cv::Point2f(i, j) - attent_pt);
             int discontinuity_counter = 0;
             int discontinuity_history[threshold];
             for (float y = 0.0f; y < dist; y += 2.0f) {
                cv::Point next_pt = cv::Point(
                   attent_pt.x + y * std::cos(angle),
                   attent_pt.y + y * std::sin(angle));
                if (mask.at<float>(next_pt.y, next_pt.x) != 255) {
                   discontinuity_history[discontinuity_counter] = 1;
                } else {
                   discontinuity_history[discontinuity_counter] = 0;
                }
                if (discontinuity_counter++ == threshold) {
                   bool is_discont = false;
                   for (int x = threshold - 1; x >= 0; x--) {
                      if (discontinuity_history[x] == 0) {
                         is_discont = false;
                         discontinuity_counter = 0;
                         break;
                      } else {
                         is_discont = true;
                      }
                   }
                   if (is_discont) {
                      // reduce the weight
                      image.at<cv::Vec3b>(j, i)[0] = 255;
                      image.at<cv::Vec3b>(j, i)[1] = 0;
                      image.at<cv::Vec3b>(j, i)[2] = 0;
                      break;
                   }
                }
             }
          }
       }
    }
    
    cv::circle(image, cv::Point(attention_pts(0), attention_pts(1)), 5,
               cv::Scalar(0, 255, 0), -1);
    
    // interpolate the image to fill small holes
    // get marked point
    // compute distance
    
    
    
    // cv::imshow("mask", mask);
    cv::imshow("image", image);
    // cv::imshow("depth", depth_map);
    cv::waitKey(3);
}

cv::Mat InteractiveSegmentation::projectPointCloudToImagePlane(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const sensor_msgs::CameraInfo::ConstPtr &camera_info,
    cv::Mat &mask, cv::Mat &indices_lut) {
    if (cloud->empty()) {
       ROS_ERROR("INPUT CLOUD EMPTY");
       return cv::Mat();
    }
    cv::Mat object_points = cv::Mat(static_cast<int>(cloud->size()), 3, CV_32F);
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_)
#endif
    for (int i = 0; i < cloud->size(); i++) {
       object_points.at<float>(i, 0) = cloud->points[i].x;
       object_points.at<float>(i, 1) = cloud->points[i].y;
       object_points.at<float>(i, 2) = cloud->points[i].z;
    }
    float K[9];
    float R[9];
    for (int i = 0; i < 9; i++) {
       K[i] = camera_info->K[i];
       R[i] = camera_info->R[i];
    }
    cv::Mat camera_matrix = cv::Mat(3, 3, CV_32F, K);
    cv::Mat rotation_matrix = cv::Mat(3, 3, CV_32F, R);
    float tvec[3];
    tvec[0] = camera_info->P[3];
    tvec[1] = camera_info->P[7];
    tvec[2] = camera_info->P[11];
    cv::Mat translation_matrix = cv::Mat(3, 1, CV_32F, tvec);

    float D[5];
    for (int i = 0; i < 5; i++) {
       D[i] = camera_info->D[i];
    }
    cv::Mat distortion_model = cv::Mat(5, 1, CV_32F, D);
    cv::Mat rvec;
    cv::Rodrigues(rotation_matrix, rvec);
    
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(object_points, rvec, translation_matrix,
                      camera_matrix, distortion_model, image_points);
    cv::Scalar color = cv::Scalar(0, 0, 0);
    cv::Mat image = cv::Mat(
       camera_info->height, camera_info->width, CV_8UC3, color);
    mask = cv::Mat::zeros(
       camera_info->height, camera_info->width, CV_32F);
    cv::Mat depth_map = cv::Mat::zeros(
       camera_info->height, camera_info->width, CV_8UC1);
    indices_lut = cv::Mat::zeros(
        camera_info->height, camera_info->width, CV_8UC1);
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_)
#endif
    for (int i = 0; i < image_points.size(); i++) {
       int x = image_points[i].x;
       int y = image_points[i].y;
       if (!isnan(x) && !isnan(y) && (x >= 0 && x <= image.cols) &&
           (y >= 0 && y <= image.rows)) {
          image.at<cv::Vec3b>(y, x)[2] = cloud->points[i].r;
          image.at<cv::Vec3b>(y, x)[1] = cloud->points[i].g;
          image.at<cv::Vec3b>(y, x)[0] = cloud->points[i].b;
          
          mask.at<float>(y, x) = 255.0f;
          depth_map.at<uchar>(y, x) = (cloud->points[i].z / 10.0f) * 255.0f;
          indices_lut.at<uchar>(y, x) = i;
       }
    }
    return image;
}

void InteractiveSegmentation::highCurvatureEdgeBoundary(
    pcl::PointCloud<PointT>::Ptr concave_edge_points,
    pcl::PointCloud<PointT>::Ptr convex_edge_points,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    const std_msgs::Header header) {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: INPUT CLOUD EMPTY FOR HIGH CURV. EST.");
       return;
    }
    *convex_edge_points = *cloud;
    *concave_edge_points = *cloud;
    
    concave_edge_points->clear();
    convex_edge_points->clear();

    pcl::PointCloud<PointT>::Ptr curv_cloud(new pcl::PointCloud<PointT>);
    *curv_cloud = *cloud;
    
    // int k = 50;  // thresholds
    // pcl::PointCloud<pcl::Normal>::Ptr normals(
    //    new pcl::PointCloud<pcl::Normal>);
    // this->estimatePointCloudNormals<int>(curv_cloud, normals, k, true);
    
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(curv_cloud);
    int search = 50;  // thresholds

    int icount = 0;
    const float concave_thresh = 0.30f;  // thresholds
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_) shared(kdtree)
#endif
    for (int i = 0; i < curv_cloud->size(); i++) {
       PointT centroid_pt = curv_cloud->points[i];
       if (!isnan(centroid_pt.x) || !isnan(centroid_pt.y) ||
           !isnan(centroid_pt.z)) {
          std::vector<int> point_idx_search;
          std::vector<float> point_squared_distance;
          // int search_out = kdtree.nearestKSearch(
          //    centroid_pt, search, point_idx_search, point_squared_distance);
          int search_out = kdtree.radiusSearch(
             centroid_pt, 0.01f, point_idx_search, point_squared_distance);
          Eigen::Vector4f seed_vector = normals->points[
             i].getNormalVector4fMap();
          float max_diff = 0.0f;
          float min_diff = FLT_MAX;
          
          int concave_sum = 0;
          for (int j = 1; j < point_idx_search.size(); j++) {
             int index = point_idx_search[j];
             Eigen::Vector4f neigh_norm = normals->points[
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
                centroid_pt.getVector4fMap(), curv_cloud->points[
                   index].getVector4fMap(), neigh_norm);
          }
          float variance = max_diff - min_diff;
          
          if (variance > concave_thresh && concave_sum <= 0) {
             centroid_pt.r = 255 * variance;
             centroid_pt.b = 0;
             centroid_pt.g = 0;
             // curv_cloud->points[i] = centroid_pt;
             // concave_edge_points->points[icount] = centroid_pt;
             concave_edge_points->push_back(centroid_pt);
          }
          if (variance > 0.10f && variance < 1.0f && concave_sum > 0) {
             centroid_pt.g = 255 * variance;
             centroid_pt.b = 0;
             centroid_pt.r = 0;
             convex_edge_points->push_back(centroid_pt);
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
        this->edgeBoundaryOutlierFiltering(concave_edge_points);
      }
#ifdef _OPENMP
#pragma omp section
#endif
      {
        this->edgeBoundaryOutlierFiltering(convex_edge_points, 0.01f, 100);
      }
    }
    ROS_INFO("\033[31m COMPLETED \033[0m");
}

bool InteractiveSegmentation::estimateAnchorPoints(
    pcl::PointCloud<PointT>::Ptr anchor_points,
    pcl::PointCloud<PointT>::Ptr convex_points,
    pcl::PointCloud<PointT>::Ptr concave_points,
    pcl::PointIndices::Ptr anchor_indices,
    const pcl::PointCloud<PointT>::Ptr original_cloud,
    const std_msgs::Header header) {
    if (anchor_points->empty()) {
       ROS_ERROR("NO BOUNDARY REGION TO SELECT POINTS");
       return false;
    }
    // NOTE: if not concave pt then r is smallest y to selected point

    std::vector<pcl::PointIndices> cluster_convx;
    std::vector<pcl::PointIndices> cluster_concv;
    std::vector<Eigen::Vector4f> convex_edge_centroids;
    std::vector<Eigen::Vector4f> concave_edge_centroids;
#ifdef _OPENMP
#pragma omp parallel sections
#endif
    {
#ifdef _OPENMP
#pragma omp section
#endif
      {
        pcl::PointIndices::Ptr prob_indices(new pcl::PointIndices);
        this->doEuclideanClustering(cluster_convx, convex_points, prob_indices);
        convex_edge_centroids = this->thinBoundaryAndComputeCentroid(
           convex_points, original_cloud, cluster_convx, cv::Scalar(0, 255, 0));
      }
#ifdef _OPENMP
#pragma omp section
#endif
      {
         pcl::PointIndices::Ptr prob_indices(new pcl::PointIndices);
         this->doEuclideanClustering(cluster_concv,
                                     concave_points, prob_indices);
         concave_edge_centroids = this->thinBoundaryAndComputeCentroid(
           concave_points, original_cloud, cluster_concv,
           cv::Scalar(0, 0, 255));
      }
    }

    std::cout << "........................................"  << "\n";
    std::cout << "CENTROID SIZE: " << concave_edge_centroids.size() << "\t"
              << convex_edge_centroids.size() << "\n";
    std::cout << "CLUSTER SIZE: " << cluster_concv.size() << "\t"
              << cluster_convx.size() << "\n";
    
    // select point in direction of normal few dist away
    if (cluster_convx.empty()) {
       // TODO:
    }
    
    float height = -FLT_MAX;
    int center_index = -1;
    Eigen::Vector4f center;
    for (int i = 0; i < concave_edge_centroids.size(); i++) {
      if (concave_edge_centroids[i](1) > height) {
        center_index = i;
        center = concave_edge_centroids[i];
      }
    }
    center(3) = 1.0f;
    
    // for selected convex mid-point
    // TODO(.): search thru each clusters in convex_edge so it can be
    // easy to extract the cluster
    
    double object_lenght_thresh = 0.50;
    double nearest_cv_dist = DBL_MAX;
    Eigen::Vector4f cc_nearest_cv_pt;
    int cc_nearest_cluster_idx = -1;
    int cc_nearest_pt_idx = -1;
    PointT cc_center_pt;
    pcl::ExtractIndices<PointT>::Ptr eifilter(new pcl::ExtractIndices<PointT>);
    eifilter->setInputCloud(original_cloud);
    for (int i = 0; i < cluster_convx.size(); i++) {
       pcl::PointIndices::Ptr region_indices(new pcl::PointIndices);
       *region_indices = cluster_convx[i];
       eifilter->setIndices(region_indices);
       pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
       eifilter->filter(*tmp_cloud);
       for (int j = 0; j < tmp_cloud->size(); j++) {
          Eigen::Vector4f cv_pt = tmp_cloud->points[j].getVector4fMap();
          if (cv_pt(1) < center(1)) {
             double d = pcl::distances::l2(cv_pt, center);
             if (d < nearest_cv_dist && d < object_lenght_thresh) {
                nearest_cv_dist = d;
                cc_nearest_cluster_idx = i;
                cc_nearest_pt_idx = j;
                cc_nearest_cv_pt = cv_pt;
                cc_center_pt = tmp_cloud->points[j];
             }
          }
       }
    }
    
    if (cc_nearest_pt_idx == -1) {
       ROS_WARN("NO NEAREST INDEX FOUND");
       return false;
    }
    
    float ap_search_radius = static_cast<float>(nearest_cv_dist)/2.0f;
    
    // find 2 points on object cloud points ap_search_radius away
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(anchor_points);
    std::vector<int> point_idx_search;
    std::vector<float> point_squared_distance;
    // PointT cc_center_pt;
    // cc_center_pt.x = cc_nearest_cv_pt(0);
    // cc_center_pt.y = cc_nearest_cv_pt(1);
    // cc_center_pt.z = cc_nearest_cv_pt(2);
    // cc_center_pt.r = 255;
    // cc_center_pt.g = 0;
    // cc_center_pt.b = 0;
    int search_out = kdtree.radiusSearch(cc_center_pt,
                                         ap_search_radius, point_idx_search,
                                         point_squared_distance);
    int ap_index_1 = -1;
    double far_point_dist = 0;
    for (int i = 0; i < point_idx_search.size(); i++) {
       Eigen::Vector4f pt = anchor_points->points[
           point_idx_search[i]].getVector4fMap();
       double d = pcl::distances::l2(cc_nearest_cv_pt, pt);
       // TODO(HERE): filter the points cross in the plane
       if (d > far_point_dist && pt(1) > cc_nearest_cv_pt(1)) {
          far_point_dist = d;
          ap_index_1 = point_idx_search[i];
          cc_nearest_cv_pt(1) = pt(1);
       }
    }
    // TODO: UNKNOWN--> complete
    if (ap_index_1 == -1) {
      ROS_ERROR("ERROR: COMPUTING THE ANCHOR POINTS");
      return false;
    }
    
    far_point_dist = 0;
    int ap_index_2 = -1;
    for (int i = 0; i < point_idx_search.size(); i++) {
       double d = pcl::distances::l2(anchor_points->points[
           point_idx_search[i]].getVector4fMap(), anchor_points->points[
               ap_index_1].getVector4fMap());
       if (d > far_point_dist) {
          far_point_dist = d;
          ap_index_2 = point_idx_search[i];
       }
    }
    
    if (ap_index_2 == -1) {
       // process with ap_index_1 only
    } else {
       // use both
    }

    // find center closest normal
    double min_dist = DBL_MAX;
    center_index = -1;
    for (int i = 0; i < anchor_points->size(); i++) {
       Eigen::Vector4f pt = anchor_points->points[i].getVector4fMap();
       double d = pcl::distances::l2(pt, cc_center_pt.getVector4fMap());
       if (d < min_dist) {
          min_dist = d;
          center_index = i;
       }
    }
    
    PointT ap_pt1 = anchor_points->points[ap_index_1];
    PointT ap_pt2 = anchor_points->points[ap_index_2];
    PointT ap_ct = anchor_points->points[center_index];
    
    anchor_points->clear();
    anchor_points->push_back(ap_pt1);
    anchor_points->push_back(ap_pt2);
    // anchor_points->push_back(cc_center_pt);
    anchor_points->push_back(ap_ct);

    anchor_indices->indices.clear();
    anchor_indices->indices.push_back(ap_index_1);
    anchor_indices->indices.push_back(ap_index_2);
    anchor_indices->indices.push_back(center_index);

    return true;
}

std::vector<Eigen::Vector4f>
InteractiveSegmentation::doEuclideanClustering(
    std::vector<pcl::PointIndices> &cluster_indices,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointIndices::Ptr prob_indices, bool is_centroid,
    const float tolerance_thresh, const int min_size_thresh,
    const int max_size_thresh) {
    cluster_indices.clear();
    std::vector<Eigen::Vector4f> cluster_centroids;
    if (cloud->empty()) {
       return cluster_centroids;
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
    euclidean_clustering.extract(cluster_indices);
    if (is_centroid) {
       pcl::ExtractIndices<PointT>::Ptr eifilter(
          new pcl::ExtractIndices<PointT>);
       eifilter->setInputCloud(cloud);
       for (int i = 0; i < cluster_indices.size(); i++) {
          pcl::PointIndices::Ptr region_indices(new pcl::PointIndices);
          *region_indices = cluster_indices[i];
          eifilter->setIndices(region_indices);
          pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
          eifilter->filter(*tmp_cloud);
          Eigen::Vector4f center;
          pcl::compute3DCentroid<PointT, float>(*tmp_cloud, center);
          cluster_centroids.push_back(center);
       }
    }
    return cluster_centroids;
}

void InteractiveSegmentation::edgeBoundaryOutlierFiltering(
    pcl::PointCloud<PointT>::Ptr cloud, const float search_radius_thresh,
    const int min_neigbor_thresh) {
    if (cloud->empty()) {
       ROS_WARN("SKIPPING OUTLIER FILTERING");
       return;
    }
    ROS_INFO("\033[32m FILTERING OUTLIER \033[0m");
    pcl::PointCloud<PointT>::Ptr concave_edge_points(
       new pcl::PointCloud<PointT>);
    pcl::RadiusOutlierRemoval<PointT>::Ptr filter_ror(
       new pcl::RadiusOutlierRemoval<PointT>);
    filter_ror->setInputCloud(cloud);
    filter_ror->setRadiusSearch(search_radius_thresh);
    filter_ror->setMinNeighborsInRadius(min_neigbor_thresh);
    filter_ror->filter(*concave_edge_points);
    cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*concave_edge_points, *cloud);
}

std::vector<Eigen::Vector4f>
InteractiveSegmentation::thinBoundaryAndComputeCentroid(
    pcl::PointCloud<PointT>::Ptr edge_cloud,
    const pcl::PointCloud<PointT>::Ptr original_cloud,
    std::vector<pcl::PointIndices> &cluster_indices, const cv::Scalar color) {
    std::vector<Eigen::Vector4f> cluster_centroids;
    pcl::PointCloud<PointT>::Ptr copy_ec(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(*edge_cloud, *copy_ec);
    pcl::ExtractIndices<PointT>::Ptr eifilter(new pcl::ExtractIndices<PointT>);
    eifilter->setInputCloud(copy_ec);
    std::vector<pcl::PointIndices> tmp_ci;
    edge_cloud->clear();
    cluster_centroids.clear();
    const int min_size_thresh = 30;
    for (int i = 0; i < cluster_indices.size(); i++) {
       pcl::PointIndices::Ptr region_indices(new pcl::PointIndices);
       *region_indices = cluster_indices[i];
       eifilter->setIndices(region_indices);
       pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
       eifilter->filter(*tmp_cloud);
       pcl::PointIndices::Ptr indices(new pcl::PointIndices);
       this->skeletonization2D(tmp_cloud, indices, original_cloud,
                               this->camera_info_, color);
       if (tmp_cloud->size() > min_size_thresh) {
          tmp_ci.push_back(*indices);
          *edge_cloud = *edge_cloud + *tmp_cloud;
          Eigen::Vector4f center;
          pcl::compute3DCentroid<PointT, float>(*tmp_cloud, center);
          cluster_centroids.push_back(center);
       }
    }
    cluster_indices.clear();
    cluster_indices.insert(cluster_indices.end(), tmp_ci.begin(), tmp_ci.end());
    return cluster_centroids;
}

bool InteractiveSegmentation::skeletonization2D(
    pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr indices,
    const pcl::PointCloud<PointT>::Ptr original_cloud,
    const sensor_msgs::CameraInfo::ConstPtr &camera_info,
    const cv::Scalar color) {
    if (cloud->empty() && original_cloud->height > 1) {
       ROS_WARN("EMPTY CLOUD SKIPPING SKELETONIZATION");
       return false;
    }
    cv::Mat mask_img;
    cv::Mat depth_image;
    mask_img = this->projectPointCloudToImagePlane(
       cloud, camera_info, mask_img, depth_image);
    cv::cvtColor(mask_img, mask_img, CV_BGR2GRAY);
    cv::threshold(mask_img, mask_img, 0, 255,
                  CV_THRESH_BINARY | CV_THRESH_OTSU);
    int erosion_size = 5;
    cv::Mat element = cv::getStructuringElement(
       cv::MORPH_ELLIPSE,
       cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
       cv::Point(erosion_size, erosion_size));
    cv::dilate(mask_img, mask_img, element);
    cv::erode(mask_img, mask_img, element);
    boost::shared_ptr<jsk_perception::Skeletonization> skeleton(
       new jsk_perception::Skeletonization());
    skeleton->skeletonization(mask_img);
    cloud->clear();
    indices->indices.clear();
    for (int j = 0; j < mask_img.rows; j++) {
       for (int i = 0; i < mask_img.cols; i++) {
          if (mask_img.at<float>(j, i) == 1.0f) {
             int index = i + (j * mask_img.cols);
             PointT pt = original_cloud->points[index];
             if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z)) {
                pt.r = color.val[2];
                pt.g = color.val[1];
                pt.b = color.val[0];
                cloud->push_back(pt);
                indices->indices.push_back(index);
             }
          }
       }
    }
    // cv_bridge::CvImagePtr pub_msg(new cv_bridge::CvImage);
    // pub_msg->header = camera_info->header;
    // pub_msg->encoding = sensor_msgs::image_encodings::MONO8;
    // pub_msg->image = mask_img.clone();
    // this->pub_image_.publish(pub_msg);
    return true;
}

void InteractiveSegmentation::publishAsROSMsg(
    const pcl::PointCloud<PointT>::Ptr cloud, const ros::Publisher publisher,
    const std_msgs::Header header) {
    if (cloud->empty()) {
      return;
    }
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = header;
    publisher.publish(ros_cloud);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "interactive_segmentation");
    InteractiveSegmentation is;
    ros::spin();
    return 0;
}
