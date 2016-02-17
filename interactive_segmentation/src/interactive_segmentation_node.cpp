// Copyright (C) 2015 by Krishneel Chaudhary, JSK Lab,
// The University of Tokyo, Japan

#include <interactive_segmentation/interactive_segmentation.h>
#include <vector>

InteractiveSegmentation::InteractiveSegmentation():
    is_init_(true), is_stop_signal_(false), num_threads_(8) {
    pnh_.getParam("num_threads", this->num_threads_);
    this->srv_ = boost::shared_ptr<dynamic_reconfigure::Server<Config> >(
       new dynamic_reconfigure::Server<Config>);
    dynamic_reconfigure::Server<Config>::CallbackType f = boost::bind(
       &InteractiveSegmentation::configCallback, this, _1, _2);
    this->srv_->setCallback(f);
    
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
    
    this->pub_apoints_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
          "/interactive_segmentation/output/anchor_points", 1);

    this->pub_normal_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/interactive_segmentation/output/normal", 1);
    
    this->pub_pt_map_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
        "/interactive_segmentation/output/point_map", 1);

    this->pub_concave_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
        "/interactive_segmentation/output/concave_edge", 1);

    this->pub_convex_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
        "/interactive_segmentation/output/convex_edge", 1);
    
    this->pub_plane_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/interactive_segmentation/output/plane_info", 1);
}

void InteractiveSegmentation::subscribe() {

       this->sub_screen_pt_.subscribe(this->pnh_, "input_screen", 1);
       this->sub_orig_cloud_.subscribe(this->pnh_, "input_orig_cloud", 1);
       this->usr_sync_ = boost::make_shared<message_filters::Synchronizer<
         UsrSyncPolicy> >(100);
       usr_sync_->connectInput(sub_screen_pt_, sub_orig_cloud_);
       usr_sync_->registerCallback(boost::bind(
           &InteractiveSegmentation::screenPointCallback, this, _1, _2));

       this->sub_polyarray_ = pnh_.subscribe(
          "/multi_plane_estimate/output_refined_polygon", 1,
          &InteractiveSegmentation::polygonArrayCallback, this);
       
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

void InteractiveSegmentation::polygonArrayCallback(
    const jsk_recognition_msgs::PolygonArray::ConstPtr &poly_msg) {
    this->polygon_array_ = *poly_msg;
}

void InteractiveSegmentation::callback(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const sensor_msgs::CameraInfo::ConstPtr &info_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::PointCloud2::ConstPtr &orig_cloud_msg) {
    if (!is_init_) {
       ROS_ERROR("ERROR: MARKED A TARGET REGION IN THE CLUSTER");
       return;
    }
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
    ROS_INFO("\033[34m ESTIMATIONG CLOUD NORMALS \033[0m");

    int k = 50;  // thresholds
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    this->estimatePointCloudNormals<int>(cloud, normals, k, true);
    ROS_INFO("\033[34m CLOUD NORMALS ESTIMATED\033[0m");

    pcl::PointCloud<PointT>::Ptr concave_edge_points(
       new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr convex_edge_points(
       new pcl::PointCloud<PointT>);
    this->highCurvatureEdgeBoundary(concave_edge_points, convex_edge_points,
                                    cloud, normals, cloud_msg->header);
    pcl::PointCloud<PointT>::Ptr anchor_points(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(*cloud, *anchor_points);
    pcl::PointIndices::Ptr anchor_indices(new pcl::PointIndices);
    pcl::PointIndices::Ptr filtered_indices(new pcl::PointIndices);
    Eigen::Vector4f plane_point = Eigen::Vector4f(0, 0, 0, 0);
    bool is_found_points = this->estimateAnchorPoints(
       anchor_points, convex_edge_points, concave_edge_points,
       anchor_indices, filtered_indices, plane_point, original_cloud);
    
    ROS_INFO("\033[32m LABELING ON THE POINT \033[0m");

    if (is_found_points) {
       pcl::PointCloud<PointT>::Ptr weight_cloud(new pcl::PointCloud<PointT>);
       this->selectedVoxelObjectHypothesis(
          weight_cloud, cloud, normals, anchor_indices, cloud_msg->header);

       // fix for index irregularity after plane segm. fix 4 imprv
       // compt.
       pcl::PointCloud<PointT>::Ptr obj_cloud(new pcl::PointCloud<PointT>);
       pcl::copyPointCloud<PointT, PointT>(*cloud, *obj_cloud);
       if (!filtered_indices->indices.empty()) {
          obj_cloud->clear();
          pcl::PointCloud<PointT>::Ptr filtered_cloud(
             new pcl::PointCloud<PointT>);
          for (int i = 0; i < filtered_indices->indices.size(); i++) {
             int idx = filtered_indices->indices[i];
             filtered_cloud->push_back(weight_cloud->points[idx]);
             obj_cloud->push_back(cloud->points[idx]);  // mask orignal region
          }
          weight_cloud->clear();
          pcl::copyPointCloud<PointT, PointT>(*filtered_cloud, *weight_cloud);
       }
       pcl::PointIndices::Ptr object_indices(new pcl::PointIndices);
       pcl::PointCloud<PointT>::Ptr final_object(new pcl::PointCloud<PointT>);
       this->attentionSurfelRegionPointCloudMask(
          weight_cloud, anchor_points->points[0].getVector4fMap(),
          final_object, object_indices);
       cloud->clear();
       for (int i = 0; i < object_indices->indices.size(); i++) {
          int idx = object_indices->indices[i];
          cloud->push_back(obj_cloud->points[idx]);
       }
       
       // check if it is the marked object --------
       if (this->markedPointInSegmentedRegion(
              final_object, this->user_marked_pt_)) {
          this->is_stop_signal_ = true;
       }
       // ----------------------------------------
       
       std::vector<pcl::PointIndices> all_indices;
       all_indices.push_back(*object_indices);
       
       jsk_recognition_msgs::ClusterPointIndices ros_indices;
       ros_indices.cluster_indices = pcl_conversions::convertToROSPointIndices(
          all_indices, cloud_msg->header);
       ros_indices.header = cloud_msg->header;
       pub_indices_.publish(ros_indices);
       
       // publishAsROSMsg(final_object, pub_cloud_, cloud_msg->header);
       publishAsROSMsg(cloud, pub_cloud_, cloud_msg->header);
    }
    
    ROS_INFO("\033[34m PUBLISHING INFO\033[0m");
    this->supportPlaneNormal(plane_point, cloud_msg->header);
    this->publishAsROSMsg(anchor_points, pub_apoints_, cloud_msg->header);
    this->publishAsROSMsg(concave_edge_points, pub_concave_, cloud_msg->header);
    this->publishAsROSMsg(convex_edge_points, pub_convex_, cloud_msg->header);

    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr centroid_normal(
       new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    for (int i = 0; i < anchor_indices->indices.size(); i++) {
       int idx = anchor_indices->indices[i];
       pcl::Normal norm = normals->points[idx];
       pcl::PointXYZRGBNormal pt;
       //! fix this as cloud is filtered avbove ----<<
       // pt.x = cloud->points[idx].x;
       // pt.y = cloud->points[idx].y;
       // pt.z = cloud->points[idx].z;
       pt.r = 255;
       pt.g = 0;
       pt.b = 255;
       pt.normal_x = norm.normal_x;
       pt.normal_y = norm.normal_y;
       pt.normal_z = norm.normal_z;
       centroid_normal->push_back(pt);
    }
    sensor_msgs::PointCloud2 ros_normal;
    pcl::toROSMsg(*centroid_normal, ros_normal);
    ros_normal.header = cloud_msg->header;
    pub_normal_.publish(ros_normal);
    
    ROS_INFO("\n\033[34m ALL VALID REGION LABELED \033[0m");
}

void InteractiveSegmentation::selectedVoxelObjectHypothesis(
    pcl::PointCloud<PointT>::Ptr weight_cloud,
    const pcl::PointCloud<PointT>::Ptr in_cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    const pcl::PointIndices::Ptr indices,
    const std_msgs::Header header) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    const int mem_size = indices->indices.size();
    std::vector<cv::Mat> anchor_points_weights(mem_size);
    std::vector<float> anchor_points_max(mem_size);

    Eigen::Vector4f attention_centroid;
    Eigen::Vector4f attention_normal;
    for (int i = 0; i < indices->indices.size(); i++) {
       cloud->clear();
       pcl::copyPointCloud<PointT, PointT>(*in_cloud, *cloud);
       int index = indices->indices[i];
       attention_normal = normals->points[index].getNormalVector4fMap();
       attention_centroid = cloud->points[index].getVector4fMap();
       
       ROS_INFO("\033[34m COMPUTING WEIGHTS \033[0m");
       
       cv::Mat weight_map;
       this->surfelSamplePointWeightMap(cloud, normals, cloud->points[index],
                                        attention_normal, weight_map);
       anchor_points_weights[i] = weight_map;

       ROS_INFO("\033[34m WEIGHTS COMPUTED\033[0m");
       // publishAsROSMsg(cloud, pub_cloud_, header);
    }
    cv::Mat conv_weight_map = cv::Mat::zeros(
       static_cast<int>(in_cloud->size()), 1, CV_32F);
    for (int i = 0; i < anchor_points_weights.size(); i++) {
       cv::Mat weight_map = anchor_points_weights[i];
       for (int j = 0; j < weight_map.rows; j++) {
          conv_weight_map.at<float>(j, 0) += (weight_map.at<float>(j, 0));
       }
       anchor_points_weights[i] = weight_map;
    }
    cv::normalize(conv_weight_map, conv_weight_map, 0, 1,
                  cv::NORM_MINMAX, -1, cv::Mat());
    pcl::copyPointCloud<PointT, PointT>(*in_cloud, *weight_cloud);
    for (int j = 0; j < conv_weight_map.rows; j++) {
       float w = (conv_weight_map.at<float>(j, 0) / 1.0f) *  255.0f;
       weight_cloud->points[j].r = w;
       weight_cloud->points[j].g = w;
       weight_cloud->points[j].b = w;
    }
    ROS_INFO("\033[34m RETURING OBJECT HYPOTHESIS \033[0m");
    // publishAsROSMsg(weight_cloud, pub_cloud_, header);
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
          connection = acos(current_normal.dot(attention_normal))/
              (1.0f * M_PI);
          // connection = std::pow((current_normal.dot(attention_normal)), 2);
       }
       // connectivity_weights.push_back(connection);
       connectivity_weights.at<float>(i, 0) = connection;
       
       Eigen::Vector3f view_point_vec = (cloud->points[i].getVector3fMap()
                                         - centroid_pt.getVector3fMap());
       // Eigen::Vector3f surface_normal_vec = normals->points[
       //    i].getNormalVector3fMap();

       Eigen::Vector3f surface_normal_vec = normals->points[
          i].getNormalVector3fMap() - attention_normal.head<3>();



       // TODO(HERE):  add Gaussian centered at selected
       float cross_norm = static_cast<float>(
           surface_normal_vec.cross(view_point_vec).norm());
       float scalar_prod = static_cast<float>(
           surface_normal_vec.dot(view_point_vec));
       float angle = atan2(cross_norm, scalar_prod);
       
       float view_pt_weight = (1.0f * CV_PI - angle)/(1.0 * CV_PI);
       // view_pt_weight = 1.0f / (1.0f + (view_pt_weight * view_pt_weight));
       // view_pt_weight = std::exp(-1.0f * view_pt_weight);


       // angle = acos(attention_normal.dot(current_normal) /
       //              (attention_normal.norm() * current_normal.norm()));
       // view_pt_weight = (angle)/(CV_PI);
       // view_pt_weight = exp(-1.5f * view_pt_weight);
       
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
     */         
     // smoothing HERE
     /*
     const int filter_lenght = 5;
     cv::GaussianBlur(connectivity_weights, connectivity_weights,
                      cv::Size(filter_lenght, filter_lenght), 0, 0);
     cv::GaussianBlur(orientation_weights, orientation_weights,
                      cv::Size(filter_lenght, filter_lenght), 0.0,
     0.0);

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

/**
 * NOT IN USE
 */
void InteractiveSegmentation::filterAndComputeNonObjectRegionAnchorPoint(
    pcl::PointCloud<PointT>::Ptr anchor_points,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    const int c_index, const cv::Mat &weight_map, const float threshold) {
    if (weight_map.empty() || anchor_points->empty() ||
        (weight_map.rows != anchor_points->size())) {
       ROS_ERROR("EMPTY CLOUD TO COMPUTE NON-OBJECT AP");
       return;
    }
    Eigen::Vector4f c_centroid = anchor_points->points[
       c_index].getVector4fMap();
    
    // TODO(HERE): RESTRICT TO ONLY TWO CONCAVE BOUNDARIES
    pcl::PointIndices::Ptr prob_indices(new pcl::PointIndices);
    pcl::PointIndices::Ptr object_indices(new pcl::PointIndices);
    pcl::PointCloud<PointT>::Ptr non_object_cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr object_cloud(new pcl::PointCloud<PointT>);
    for (int i = 0; i < weight_map.rows; i++) {
       float weight = weight_map.at<float>(i, 0);
       if (weight == threshold && anchor_points->points[i].y > c_centroid(1)) {
          non_object_cloud->push_back(anchor_points->points[i]);
          prob_indices->indices.push_back(i);
       } else {
         object_cloud->push_back(anchor_points->points[i]);
         object_indices->indices.push_back(i);
       }
    }
    std::vector<pcl::PointIndices> cluster_indices;
    std::vector<Eigen::Vector4f> cluster_centroids;
    cluster_centroids = this->doEuclideanClustering(
       cluster_indices, anchor_points, prob_indices, true, 0.01f, 100);

    double distance = DBL_MAX;
    int index = -1;
    for (int i = 0; i < cluster_centroids.size(); i++) {
       double d = pcl::distances::l2(c_centroid, cluster_centroids[i]);
       if (d < distance) {
          distance = d;
          index = i;
       }
    }
    distance = DBL_MAX;
    int idx = -1;
    for (int i = 0; i < cluster_indices[index].indices.size(); i++) {
      int j = cluster_indices[index].indices[i];
      double d = pcl::distances::l2(cluster_centroids[index],
                                    anchor_points->points[j].getVector4fMap());
      if (d < distance) {
        distance = d;
        idx = j;
      }
    }
    Eigen::Vector4f non_obj_ap_pt = anchor_points->points[idx].getVector4fMap();
    Eigen::Vector4f cc_normal = normals->points[c_index].getNormalVector4fMap();

    // cv::Mat weights;
    // this->surfelSamplePointWeightMap(anchor_points, normals,
    // anchor_points->points[idx],
    // normals->points[idx].getNormalVector4fMap(), weights);
    

    for (int i = 0; i < anchor_points->size(); i++) {
      // anchor_points->points[i].r = weights.at<float>(i, 0) * 255.0f;
      // anchor_points->points[i].b = weights.at<float>(i, 0) * 255.0f;
      // anchor_points->points[i].g = weights.at<float>(i, 0) * 255.0f;
      int ind = i;
      int val = this->localVoxelConvexityCriteria(
          non_obj_ap_pt, anchor_points->points[ind].getVector4fMap(),
          normals->points[ind].getNormalVector4fMap());
      if (val == -1) {
        val = 0;
      }
      float pix_val = static_cast<float>(val) *
         weight_map.at<float>(i, 0) * 255.0f;

      anchor_points->points[ind].r = pix_val;
      anchor_points->points[ind].b = pix_val;
      anchor_points->points[ind].g = pix_val;
    }
    // this->publishAsROSMsg(anchor_points, pub_prob_, camera_info_->header);
    this->publishAsROSMsg(non_object_cloud, pub_prob_, camera_info_->header);
    
}

bool InteractiveSegmentation::attentionSurfelRegionPointCloudMask(
    const pcl::PointCloud<PointT>::Ptr weight_cloud,
    const Eigen::Vector4f centroid,
    pcl::PointCloud<PointT>::Ptr prob_object_cloud,
    pcl::PointIndices::Ptr prob_object_indices) {
    if (weight_cloud->empty()) {
      return false;
    }
    // removed zero points
    pcl::PointIndices::Ptr prob_indices(new pcl::PointIndices);
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    const float threshold = 0.2f * 255.0f;
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
    ros_cloud.header = camera_info_->header;
    this->pub_prob_.publish(ros_cloud);
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
          curvature_ptr[i] = (eigen_vals(0) / sum_eigen);
          double dist = pcl::distances::l2(
             centroid_pt.getVector4fMap(), center);
          if (dist < distance) {
             index = i;
          }
       }
    }

    
    new_cloud->clear();
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
    int search = 100;  // thresholds

    int icount = 0;
    const float concave_thresh = 0.20f;  // thresholds
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
          if (variance > 0.20f /*&& variance < 1.0f*/ && concave_sum > 0) {
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
          double ccof = this->outlier_concave_;
          this->edgeBoundaryOutlierFiltering(concave_edge_points,
                                             static_cast<float>(ccof));
       }
#ifdef _OPENMP
#pragma omp section
#endif
       {
          double cvof = this->outlier_convex_;
          this->pnh_.getParam("outlier_convex", cvof);
          this->edgeBoundaryOutlierFiltering(convex_edge_points,
                                             static_cast<float>(cvof));
       }
    }
    this->publishAsROSMsg(concave_edge_points, pub_concave_, header);
    this->publishAsROSMsg(convex_edge_points, pub_convex_, header);
    ROS_INFO("\033[31m COMPLETED \033[0m");
}

bool InteractiveSegmentation::estimateAnchorPoints(
    pcl::PointCloud<PointT>::Ptr anchor_points,
    pcl::PointCloud<PointT>::Ptr convex_points,
    pcl::PointCloud<PointT>::Ptr concave_points,
    pcl::PointIndices::Ptr anchor_indices,
    pcl::PointIndices::Ptr filter_indices, Eigen::Vector4f &center,
    const pcl::PointCloud<PointT>::Ptr original_cloud) {
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


    // TODO(here): if only convex then chouse the top most

    if (cluster_convx.size() == 1 && cluster_concv.empty()) {
       double dist = DBL_MAX;
       int indx = -1;
       PointT pt;
       for (int i = 0; i < cluster_convx[0].indices.size(); i++) {
          int idx = cluster_convx[0].indices[i];
          pt = original_cloud->points[idx];
          double d = pcl::distances::l2(convex_edge_centroids[0],
                                        pt.getVector4fMap());
          if (d < dist) {
             dist = d;
             indx = idx;
          }
       }
       if (indx != -1) {
          anchor_points->clear();
          anchor_points->push_back(pt);
          anchor_indices->indices.clear();
          anchor_indices->indices.push_back(indx);
          return true;
       } else {
          return false;
       }
    }
    
    if (cluster_concv.empty() ||
        (cluster_concv.empty() && cluster_convx.empty())) {
       ROS_ERROR("RETURNING CLOUD CENTROID");
       // return cloud centroid
       Eigen::Vector4f centroid;
       pcl::compute3DCentroid<PointT, float>(*anchor_points, centroid);
       double dist = DBL_MAX;
       int indx = -1;
       for (int i = 0; i < anchor_points->size(); i++) {
          double d = pcl::distances::l2(
             centroid, anchor_points->points[i].getVector4fMap());
          if (d < dist) {
             dist = d;
             indx = i;
          }
       }
       if (indx == -1) {
          return false;
       } else {
          PointT pt = anchor_points->points[indx];
          anchor_points->clear();
          anchor_points->push_back(pt);
          anchor_indices->indices.push_back(indx);
          return true;
       }
    }
    
    float height = FLT_MAX;
    int center_index = -1;
    for (int i = 0; i < concave_edge_centroids.size(); i++) {
      if (concave_edge_centroids[i](1) < height) {  // CHANGE HERE
        center_index = i;
        center = concave_edge_centroids[i];
        height = concave_edge_centroids[i](1);
      }
    }

    if (center_index == -1) {
       ROS_ERROR("ERROR: NO CONCAVE CENTER FOUND");
       return false;
    }
    
    // TODO(HERE): select closest and best candidate
    // select point in direction of normal few dist away
    pcl::ExtractIndices<PointT>::Ptr eifilter(new pcl::ExtractIndices<PointT>);
    eifilter->setInputCloud(original_cloud);
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(anchor_points);
    std::vector<int> point_idx_search;
    std::vector<float> point_squared_distance;
    if (cluster_convx.empty()) {
       pcl::PointIndices::Ptr region_indices(new pcl::PointIndices);
       *region_indices = cluster_concv[center_index];
       eifilter->setIndices(region_indices);
       pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
       eifilter->filter(*tmp_cloud);
       double dist = 0.0;
       center_index = -1;
       for (int i = 0; i < tmp_cloud->size(); i++) {
          double d = pcl::distances::l2(tmp_cloud->points[i].getVector4fMap(),
                                        center);
          if (d > dist) {
             dist = d;
             center_index = i;
          }
       }
       PointT center_pt;
       center_pt.x = center(0);
       center_pt.y = center(1);
       center_pt.z = center(2);
       center_pt.r = 255;
       float search_rad = static_cast<float>(dist) / 2.0f;
       int search_out = kdtree.radiusSearch(center_pt,
                                            search_rad, point_idx_search,
                                            point_squared_distance);
       if (!search_out) {
          ROS_ERROR("CONVEX EMPTY AND SEARCH FAILED");
          return false;
       }
       double far_point_dist = 0.0;
       int fp_indx = -1;
       float higher_y = center(1);
       for (int i = 0; i < point_idx_search.size(); i++) {
          Eigen::Vector4f pt = anchor_points->points[
             point_idx_search[i]].getVector4fMap();
          double d = pcl::distances::l2(center, pt);
          if (d > far_point_dist && pt(1) > higher_y) {
             far_point_dist = d;
             fp_indx = point_idx_search[i];
             higher_y = pt(1);
          }
       }
       PointT pt = anchor_points->points[fp_indx];
       anchor_points->clear();
       anchor_points->push_back(pt);
       anchor_indices->indices.push_back(fp_indx);
       return true;
    }
    
    Eigen::Vector3f polygon_normal;
    pcl::PointCloud<PointT>::Ptr truncated_points(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(*anchor_points, *truncated_points);
    this->fixPlaneModelToEdgeBoundaryPoints(truncated_points, filter_indices,
                                            polygon_normal, center);

    // No other edge above the concave edge
    if (concave_edge_centroids.size() == 1) {
       bool is_edge_above = false;
       for (int i = 0; i < convex_edge_centroids.size(); i++) {
          if (convex_edge_centroids[i](1) < center(1)) {
             is_edge_above = true;
             break;
          }
       }
       if (!is_edge_above) {
          const float fix_distance = 0.04f;
          int indx = -1;
          PointT pt;
          for (int i = 0; i < truncated_points->size(); i++) {
             pt = truncated_points->points[i];
             double d = pcl::distances::l2(pt.getVector4fMap(), center);
             if ((static_cast<float>(d) > fix_distance) &&
                 (static_cast<float>(d) < fix_distance + 0.01f) &&
                 pt.y < center(1)) {
                indx = i;
                break;
             }
          }
          if (indx != -1) {
             anchor_points->clear();
             anchor_points->push_back(pt);
             anchor_indices->indices.clear();
             anchor_indices->indices.push_back(indx);
             return true;
          } else {
             return false;
          }
       }
    }
    // -----------------------------
    
    double object_lenght_thresh = 0.50;
    double nearest_cv_dist = DBL_MAX;
    Eigen::Vector4f cc_nearest_cv_pt;
    int cc_nearest_cluster_idx = -1;
    int cc_nearest_pt_idx = -1;
    PointT cc_center_pt;
    double intra_convx_dist = 0.0;

    double cc_nearest_dist_cv_bt = DBL_MAX;
    int cc_nearest_bt_cluster_idx = -1;

    // hack to correct ap selection
    Eigen::Vector3f adj_center = center.head<3>();
    adj_center(1) -= 0.01f;
    for (int i = 0; i < cluster_convx.size(); i++) {
       float pt_pos = polygon_normal.dot(convex_edge_centroids[i].head<3>() -
                                         adj_center);
       if (pt_pos > 0.0f) {
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
             // convx intra distance
             double d = pcl::distances::l2(
                convex_edge_centroids[i],
                tmp_cloud->points[j].getVector4fMap());
             if (d > intra_convx_dist) {
                intra_convx_dist = d;
             }

             //-----------------------------
             if (cv_pt(1) > center(1)) {
                double d = pcl::distances::l2(cv_pt, center);
                if (d < cc_nearest_dist_cv_bt && d < object_lenght_thresh) {
                   cc_nearest_dist_cv_bt = d;
                   cc_nearest_bt_cluster_idx = i;
                }
             }
             //----------------------------
          }
       }
    }
    
    if (cc_nearest_pt_idx == -1) {
       ROS_WARN("NO NEAREST INDEX FOUND");
       return false;
    }
    
    // float ap_search_radius = static_cast<float>(nearest_cv_dist)/2.0f;
    float ap_search_radius = static_cast<float>(
       std::min(nearest_cv_dist, intra_convx_dist))/2.0f;


    std::cout << "DISTANCE: " << ap_search_radius << "\n";
    
    // find 2 points on object cloud points ap_search_radius away
    point_idx_search.clear();
    point_squared_distance.clear();
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
    
    if (ap_index_2 == -1) {
       PointT ap_pt1 = anchor_points->points[ap_index_1];
       PointT ap_ct = anchor_points->points[center_index];
       
       anchor_points->clear();
       anchor_points->push_back(ap_pt1);
       anchor_points->push_back(ap_ct);

       anchor_indices->indices.clear();
       anchor_indices->indices.push_back(ap_index_1);
       anchor_indices->indices.push_back(center_index);
    } else {
       std::cout << "\n 3 ANCHOR POINTS ESTIMATED" << std::endl;
       PointT ap_pt1 = anchor_points->points[ap_index_1];
       PointT ap_pt2 = anchor_points->points[ap_index_2];
       PointT ap_ct = anchor_points->points[center_index];
       
       anchor_points->clear();
       anchor_points->push_back(ap_pt1);
       anchor_points->push_back(ap_pt2);
       anchor_points->push_back(ap_ct);

       anchor_indices->indices.clear();
       anchor_indices->indices.push_back(ap_index_1);
       anchor_indices->indices.push_back(ap_index_2);
       anchor_indices->indices.push_back(center_index);
    }
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
    
    for (int i = 0; i < cluster_indices.size(); i++) {
       pcl::PointIndices::Ptr region_indices(new pcl::PointIndices);
       *region_indices = cluster_indices[i];
       eifilter->setIndices(region_indices);
       pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
       eifilter->filter(*tmp_cloud);
       pcl::PointIndices::Ptr indices(new pcl::PointIndices);
       this->skeletonization2D(tmp_cloud, indices, original_cloud,
                               this->camera_info_, color);
       if (tmp_cloud->size() > this->skeleton_min_thresh_) {
          tmp_ci.push_back(*indices);
          *edge_cloud = *edge_cloud + *tmp_cloud;
          Eigen::Vector4f center;
          pcl::compute3DCentroid<PointT, float>(*tmp_cloud, center);
          center(3) = 1.0f;
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

void InteractiveSegmentation::supportPlaneNormal(
    const Eigen::Vector4f plane_point, const std_msgs::Header header) {
    geometry_msgs::PolygonStamped polygon = polygon_array_.polygons[0];
    jsk_recognition_utils::Polygon geo_polygon
       = jsk_recognition_utils::Polygon::fromROSMsg(polygon.polygon);
    jsk_recognition_utils::Vertices vertices = geo_polygon.getVertices();
    Eigen::Vector3f normal = geo_polygon.getNormal();
    pcl::PointNormal pt;
    pt.x = plane_point(0);
    pt.y = plane_point(1);
    pt.z = plane_point(2);
    pt.normal_x = normal(0);
    pt.normal_y = normal(1);
    pt.normal_z = normal(2);
    pcl::PointCloud<pcl::PointNormal>::Ptr plane_info(
       new pcl::PointCloud<pcl::PointNormal>);
    plane_info->push_back(pt);
    sensor_msgs::PointCloud2 ros_plane;
    pcl::toROSMsg(*plane_info, ros_plane);
    ros_plane.header = header;
    this->pub_plane_.publish(ros_plane);
}

void InteractiveSegmentation::fixPlaneModelToEdgeBoundaryPoints(
    pcl::PointCloud<PointT>::Ptr in_cloud, pcl::PointIndices::Ptr indices,
    Eigen::Vector3f &normal, const Eigen::Vector4f m_centroid) {
    if (in_cloud->empty()) {
       return;
    }
    ROS_INFO("FITTING PLANE");
        
    Eigen::Vector4f center = m_centroid;
    center(1) -= 0.01f;
    geometry_msgs::PolygonStamped polygon = polygon_array_.polygons[0];
    jsk_recognition_utils::Polygon geo_polygon
       = jsk_recognition_utils::Polygon::fromROSMsg(polygon.polygon);
    jsk_recognition_utils::Vertices vertices = geo_polygon.getVertices();
    Eigen::Vector3f centroid(0, 0, 0);
    if (vertices.size() == 0) {
       ROS_ERROR("the size of vertices is 0");
    } else {
       for (size_t j = 0; j < vertices.size(); j++) {
          centroid = vertices[j] + centroid;
       }
       centroid = centroid / vertices.size();
    }
    Eigen::Vector3f pos(centroid[0], centroid[1], centroid[2]);
    normal = geo_polygon.getNormal();

    ROS_INFO("NORMAL ESTIMATED");
        
    pcl::PointCloud<PointT>::Ptr object_cloud(new pcl::PointCloud<PointT>);
    for (int i = 0; i < in_cloud->size(); i++) {
       Eigen::Vector3f pt = in_cloud->points[i].getVector3fMap();
       if (normal.dot(pt - center.head<3>()) >= 0.0f) {
          object_cloud->push_back(in_cloud->points[i]);
          indices->indices.push_back(i);
       }
    }
    in_cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*object_cloud, *in_cloud);
    this->publishAsROSMsg(object_cloud, pub_prob_, camera_info_->header);

    ROS_INFO("DONE FITTING PLANE");

    bool is_viz = false;
    if (is_viz) {
       float coef = normal.dot(center.head<3>());
       float x = coef / normal(0);
       float y = coef / normal(1);
       float z = coef / normal(2);
       Eigen::Vector3f point_x = Eigen::Vector3f(x, 0.0f, 0.0f);
       Eigen::Vector3f point_y = Eigen::Vector3f(0.0f, y, 0.0f) - point_x;
       Eigen::Vector3f point_z = Eigen::Vector3f(0.0f, 0.0f, z) - point_x;
       object_cloud->clear();
       for (float y = -1.0f; y < 1.0f; y += 0.01f) {
          for (float x = -1.0f; x < 1.0f; x += 0.01f) {
             PointT pt;
             pt.x = point_x(0) + point_y(0) * x + point_z(0) * y;
             pt.y = point_x(1) + point_y(1) * x + point_z(1) * y;
             pt.z = point_x(2) + point_y(2) * x + point_z(2) * y;
             pt.g = 255;
             object_cloud->push_back(pt);
          }
       }
       pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr centroid_normal(
          new pcl::PointCloud<pcl::PointXYZRGBNormal>);
       pcl::PointXYZRGBNormal pt;
       pt.x = pos(0);
       pt.y = pos(1);
       pt.z = pos(2);
       pt.r = 255;
       pt.g = 0;
       pt.b = 255;
       pt.normal_x = normal(0);
       pt.normal_y = normal(1);
       pt.normal_z = normal(2);
       centroid_normal->push_back(pt);
       centroid_normal->push_back(pt);
       centroid_normal->push_back(pt);
       sensor_msgs::PointCloud2 ros_normal;
       pcl::toROSMsg(*centroid_normal, ros_normal);
       ros_normal.header = camera_info_->header;
       pub_normal_.publish(ros_normal);
    }
}

bool InteractiveSegmentation::markedPointInSegmentedRegion(
    const pcl::PointCloud<PointT>::Ptr cloud, const PointT mark_pt) {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: EMPTY CLOUD FOR MARKED POINT TEST");
       return false;
    }
    Eigen::Vector4f mark = mark_pt.getVector4fMap();
    bool is_inside = false;
    for (int i = 0; i < cloud->size(); i++) {
       double d = pcl::distances::l2(cloud->points[i].getVector4fMap(), mark);
       if (d < 0.01) {
          is_inside = true;
          break;
       }
    }
    return is_inside;
}

/**
 * NOT USED
 */
void InteractiveSegmentation::normalizedCurvatureNormalHistogram(
    pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<pcl::Normal>::Ptr normals) {
    if (normals->size() != cloud->size()) {
       return;
    }
    float min_val[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float max_val[3] = {FLT_MIN, FLT_MIN, FLT_MIN};
    for (int i = 0; i < normals->size(); i++) {
       float x = normals->points[i].normal_x;
       float y = normals->points[i].normal_y;
       float z = normals->points[i].normal_z;
       max_val[0] = std::max(x, max_val[0]);
       min_val[0] = std::min(x, min_val[0]);
       max_val[1] = std::max(y, max_val[1]);
       min_val[1] = std::min(y, min_val[1]);
       max_val[2] = std::max(z, max_val[2]);
       min_val[2] = std::min(z, min_val[2]);
    }
    const int bin_size = 9;
    
    for (int i = 0; i < normals->size(); i++) {
       float x = (normals->points[i].normal_x - min_val[0]) /
          (max_val[0] - min_val[0]);
       float y = (normals->points[i].normal_y - min_val[1]) /
          (max_val[1] - min_val[1]);
       float z = (normals->points[i].normal_z - min_val[2]) /
          (max_val[2] - min_val[2]);
    }
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

void InteractiveSegmentation::configCallback(
    Config &config, uint32_t level) {
    boost::mutex::scoped_lock lock(mutex_);
    this->min_cluster_size_ = config.min_cluster_size;
    this->outlier_concave_ = config.outlier_concave;
    this->outlier_convex_ = config.outlier_convex;
    this->skeleton_min_thresh_ = config.skeleton_min_thresh;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "interactive_segmentation");
    InteractiveSegmentation is;
    ros::spin();
    return 0;
}
