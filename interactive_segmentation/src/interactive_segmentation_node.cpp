// Copyright (C) 2015 by Krishneel Chaudhary, JSK Lab,
// The University of Tokyo, Japan

#include <interactive_segmentation/interactive_segmentation.h>
#include <vector>

InteractiveSegmentation::InteractiveSegmentation():
    min_cluster_size_(100), is_init_(true),
    num_threads_(8) {
    pnh_.getParam("num_threads", this->num_threads_);

    this->srv_client_ = this->pnh_.serviceClient<
      interactive_segmentation::OutlierFiltering>("outlier_filtering_srv");
    
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
          "/interactive_segmentation/output/supervoxels", 1);

    // this->pub_normal_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
    //    "/interactive_segmentation/output/normal", sizeof(char));
    
    this->pub_pt_map_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
        "/interactive_segmentation/output/point_map", 1);
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
    const sensor_msgs::PointCloud2::ConstPtr &normal_msg) {
    boost::mutex::scoped_lock lock(this->mutex_);
    cv::Mat image = cv_bridge::toCvShare(
       image_msg, image_msg->encoding)->image;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
        
    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud<PointT>(*cloud, *cloud, nan_indices);

    ROS_INFO("\033[32m DEBUG: PROCESSING CALLBACK \033[0m");
    
    // ----------------------------------------

    this->highCurvatureConcaveBoundary(cloud, cloud, cloud_msg->header);
    cv::Mat mask_img;
    cv::Mat depth_img;
    mask_img = this->projectPointCloudToImagePlane(cloud, info_msg,
                                                   mask_img, depth_img);
    cv_bridge::CvImagePtr pub_msg(new cv_bridge::CvImage);
    pub_msg->header = info_msg->header;
    pub_msg->encoding = sensor_msgs::image_encodings::BGR8;
    pub_msg->image = mask_img.clone();
    this->pub_image_.publish(pub_msg);
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
    
    return;
    // ----------------------------------------
    
    bool is_surfel_level = true;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
    if (is_surfel_level) {
       this->surfelLevelObjectHypothesis(
          cloud, normals, supervoxel_clusters);
       sensor_msgs::PointCloud2 ros_voxels;
       jsk_recognition_msgs::ClusterPointIndices ros_indices;
       this->publishSupervoxel(supervoxel_clusters,
                               ros_voxels, ros_indices, cloud_msg->header);
       this->pub_voxels_.publish(ros_voxels);
       this->pub_indices_.publish(ros_indices);
    }
  

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr centroid_cloud(
       new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::Normal>::Ptr surfel_normals(
       new pcl::PointCloud<pcl::Normal>);


    // align supervoxel centroid to current point cloud
    const int sv_size = static_cast<int>(supervoxel_clusters.size());
    std::vector<int> aligned_indices(sv_size);
    int *a_indices = &aligned_indices[0];
    std::vector<uint32_t> supervoxel_index(sv_size);
    uint32_t *sv_index = &supervoxel_index[0];
    bool flag_bit[sv_size];
    int icount = 0;
    for (std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr >::iterator it =
            supervoxel_clusters.begin(); it != supervoxel_clusters.end();
         it++) {
       Eigen::Vector4f cur_centroid = supervoxel_clusters.at(
          it->first)->centroid_.getVector4fMap();
       double distance = DBL_MAX;
       int ind = -1;
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_) \
    shared(distance, ind)
#endif
       for (int i = 0; i < cloud->size(); i++) {
          Eigen::Vector4f cloud_pt = cloud->points[i].getVector4fMap();
          double dist = pcl::distances::l2(cur_centroid, cloud_pt);
          if (dist < distance) {
#ifdef _OPENMP
#pragma omp critical
#endif
             {
                distance = dist;
                ind = i;
             }
          }
       }
       a_indices[icount] = ind;
       sv_index[icount] = it->first;
       flag_bit[icount] = false;
       icount++;
    }
    
    pcl::PointCloud<PointT>::Ptr in_cloud(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(*cloud, *in_cloud);
    pcl::PointCloud<PointT>::Ptr non_object_cloud(new pcl::PointCloud<PointT>);


    std::cout << "\033[33m # of SuperVoxels: \033[0m"  << sv_size << std::endl;

    for (int i = 0; i < sv_size; i++) {
       if (supervoxel_clusters.at(supervoxel_index[i])->voxels_->size() >
           this->min_cluster_size_ && !flag_bit[i]) {
          pcl::PointIndices::Ptr prob_object_indices(new pcl::PointIndices);
          this->selectedVoxelObjectHypothesis(prob_object_indices,
                                              supervoxel_clusters,
                                              supervoxel_index[i],
                                              cloud, info_msg);
          non_object_cloud->clear();
          pcl::copyPointCloud<PointT, PointT>(*in_cloud,
                                              *non_object_cloud);
          for (int j = 0; j < prob_object_indices->indices.size(); j++) {
             int idx = prob_object_indices->indices[j];
             PointT pt = in_cloud->points[idx];
             pt.x = std::numeric_limits<float>::quiet_NaN();
             pt.y = std::numeric_limits<float>::quiet_NaN();
             pt.z = std::numeric_limits<float>::quiet_NaN();
             non_object_cloud->points[idx] = pt;
             for (int k = 0; k < sv_size; k++) {
                if (idx == a_indices[k]) {
                   flag_bit[k] = true;
                }
             }
          }
          cloud->clear();
          cloud->resize(non_object_cloud->size());
          for (int k = 0; k < non_object_cloud->size(); k++) {
             PointT noc_pt = non_object_cloud->points[k];
             if (!isnan(noc_pt.x) || !isnan(noc_pt.y) || !isnan(noc_pt.z)) {
                // cloud->push_back(noc_pt);
                cloud->points[k] = noc_pt;
             }
          }
          
          sensor_msgs::PointCloud2 ros_cloud;
          pcl::toROSMsg(*cloud, ros_cloud);
          ros_cloud.header = cloud_msg->header;
          this->pub_cloud_.publish(ros_cloud);
          
       } else {
          ROS_INFO("\033[32m SKIPPPED %d \033[0m", supervoxel_index[i]);
       }
       // centroid_cloud->push_back(supervoxel_clusters.at(
       //                              it->first)->centroid_);
       // *surfel_normals += *(supervoxel_clusters.at(it->first)->normals_);
    }
    
    std::cout << "\n\033[34m 1) ALL VALID REGION LABELED \033[0m" << std::endl;
    ros::Duration(10).sleep();
    /*
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
    */
}

void InteractiveSegmentation::selectedVoxelObjectHypothesis(
    pcl::PointIndices::Ptr prob_object_indices,
    const std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters,
    const uint32_t closest_surfel_index, pcl::PointCloud<PointT>::Ptr cloud,
    const sensor_msgs::CameraInfo::ConstPtr &info_msg) {
    pcl::PointCloud<PointT>::Ptr object_points(new pcl::PointCloud<PointT>);
    *object_points = *cloud;
    bool is_point_level = true;
    if (is_point_level && !supervoxel_clusters.empty()) {
       pcl::PointIndices sample_point_indices;
       // index of the sampled surfel points
       int sample_index = 0;
       sample_point_indices.indices.push_back(sample_index);
       if (is_init_) {
          pcl::PointXYZRGBA centroid_pt = supervoxel_clusters.at(
             closest_surfel_index)->centroid_;
          if (isnan(centroid_pt.x) || isnan(centroid_pt.y) ||
              isnan(centroid_pt.z)) {
             return;
          }
          
          // just use the origin cloud and norm
          int k = 100;
          pcl::PointCloud<pcl::Normal>::Ptr normals(
             new pcl::PointCloud<pcl::Normal>);
          this->estimatePointCloudNormals<int>(cloud, normals, k, true);

          
                   
          Eigen::Vector4f attention_normal = this->cloudMeanNormal(
             supervoxel_clusters.at(closest_surfel_index)->normals_);
          Eigen::Vector4f attention_centroid = centroid_pt.getVector4fMap();
         
          // select few points close to centroid as true object

          std::cout << "\033[34m 2) COMPUTING WEIGHTS \033[0m" << std::endl;
          int index_pos = 0;
          cv::Mat weight_map;
          this->surfelSamplePointWeightMap(cloud, normals, centroid_pt,
                                           attention_normal,
                                           weight_map);

          /*
          for (int i = 0; i < point_idx_search.size(); i++) {
             int idx = point_idx_search[i];
             pcl::PointXYZRGBA neigh_pt = centroid_cloud->points[idx];
             PointT obj_pt;
             obj_pt.x = neigh_pt.x;
             obj_pt.y = neigh_pt.y;
             obj_pt.z = neigh_pt.z;
             obj_pt.r = neigh_pt.r;
             obj_pt.g = neigh_pt.g;
             obj_pt.b = neigh_pt.b;
             object_points->push_back(obj_pt);

          
             // cv::Mat sample_weight_map;
             // Eigen::Vector4f idx_attn_normal = surfel_normals->points[
             //     idx].getNormalVector4fMap();
             // this->surfelSamplePointWeightMap(cloud, normals, neigh_pt,
             //                                  sample_weight_map);
             // cv::Mat tmp;
             // cv::add(weight_map, sample_weight_map, tmp);
             // weight_map = tmp.clone();
           
          }
          cv::normalize(weight_map, weight_map, 0, 1,
                        cv::NORM_MINMAX, -1, cv::Mat());
          */
         
          // normalize weights **REMOVE THIS CLOUD**
          pcl::PointCloud<PointT>::Ptr weight_cloud(
             new pcl::PointCloud<PointT>);
          for (int x = 0; x < weight_map.rows; x++) {
             cloud->points[x].r = weight_map.at<float>(x, 0) * 255.0f;
             cloud->points[x].g = weight_map.at<float>(x, 0) * 255.0f;
             cloud->points[x].b = weight_map.at<float>(x, 0) * 255.0f;
          }
          *weight_cloud = *cloud;
         
          std::cout << cloud->size() << "\t" << normals->size() << "\t"
                    << weight_cloud->size() << "\n";
          
          std::cout << "\033[34m 3) OBJECT MASK EXTRACTION \033[0m"
                    << std::endl;
        
          // weights for graph cut
          /*
          cv::Mat conv_weights = cv::Mat(image.size(), CV_32F);
          for (int i = 0; i < image.rows; i++) {
             for (int j = 0; j < image.cols; j++) {
                int idx = j + (i * image.cols);
                conv_weights.at<float>(i, j) =
                   weight_cloud->points[idx].r/255.0f;
                if (isnan(conv_weights.at<float>(i, j))) {
                   conv_weights.at<float>(i, j) = 0.0f;
                }
             }
          }
          */
        
             // get indices of the probable object mask
          pcl::PointCloud<PointT>::Ptr prob_object_cloud(
             new pcl::PointCloud<PointT>);
          this->attentionSurfelRegionPointCloudMask(cloud, attention_centroid,
                                                    info_msg->header,
                                                    prob_object_cloud,
                                                    prob_object_indices);

       
          // TMP
          /*
            pcl::PointCloud<PointT>::Ptr cov_cloud(new pcl::PointCloud<PointT>);
            this->computePointCloudCovarianceMatrix(prob_object_cloud, cov_cloud);
            cloud->clear();
            *cloud = *cov_cloud;
            */


          Eigen::Vector3f attent_pt = Eigen::Vector3f(
             this->screen_pt_.x, this->screen_pt_.y, 0);
          this->selectedPointToRegionDistanceWeight(
             prob_object_cloud, attent_pt, 0.01f, info_msg);

          pcl::PointCloud<PointT>::Ptr high_curvature_filterd(
             new pcl::PointCloud<PointT>);
          // TODO(HERE):  filter the normal and feed it in
          this->highCurvatureConcaveBoundary(high_curvature_filterd,
                                             prob_object_cloud,
                                             info_msg->header);
          
          std::cout << cloud->size() << "\t" << normals->size() << std::endl;
       }
       // cv_bridge::CvImage pub_img(
       //    image_msg->header, sensor_msgs::image_encodings::BGR8, image);
       // this->pub_image_.publish(pub_img.toImageMsg());
    }
}


void InteractiveSegmentation::surfelSamplePointWeightMap(
     const pcl::PointCloud<PointT>::Ptr cloud,
     const pcl::PointCloud<pcl::Normal>::Ptr normals,
     const pcl::PointXYZRGBA &centroid_pt,
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
    cv::Mat &mask, cv::Mat &depth_map) {
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
    depth_map = cv::Mat::zeros(
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
       }
    }
    return image;
}

void InteractiveSegmentation::highCurvatureConcaveBoundary(
    pcl::PointCloud<PointT>::Ptr filtered_cloud,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const std_msgs::Header header) {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: INPUT CLOUD EMPTY FOR HIGH CURV. EST.");
       return;
    }
    std::vector<int> indices;
    pcl::PointCloud<PointT>::Ptr curv_cloud(new pcl::PointCloud<PointT>);
    pcl::removeNaNFromPointCloud(*cloud, *curv_cloud, indices);
    
    filtered_cloud->clear();
    // filtered_cloud->resize(static_cast<int>(curv_cloud->size()));
    
    int k = 50;  // thresholds
    pcl::PointCloud<pcl::Normal>::Ptr normals(
       new pcl::PointCloud<pcl::Normal>);
    this->estimatePointCloudNormals<int>(curv_cloud, normals, k, true);

    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(curv_cloud);
    int search = 50;  // thresholds

    int icount = 0;
    const float concave_thresh = 0.30f;  // thresholds
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_) shared(kdtree, icount)
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
          // if (variance > 0.10 && variance < 1.0f && concave_sum > 0) {
             centroid_pt.r = 255 * variance;
             centroid_pt.b = 0;
             centroid_pt.g = 0;
             curv_cloud->points[i] = centroid_pt;
             // filtered_cloud->points[icount] = centroid_pt;
             filtered_cloud->push_back(centroid_pt);  // order is not import.
// #ifdef _OPENMP
// #pragma omp atomic
// #endif
             icount++;
          } else {
            // filtered_cloud->points[icount++] = centroid_pt;
          }
/*
          if (variance > 0.10 && variance < 1.0f && concave_sum > 0) {
             centroid_pt.g = 255 * variance;
             centroid_pt.b = 0;
             centroid_pt.r = 0;
             curv_cloud->points[i] = centroid_pt;
             filtered_cloud->push_back(centroid_pt);  // order is not import.
          }
          */
       }
    }

    curv_cloud->clear();
    // *curv_cloud = *filtered_cloud;
    // clear oversize memory
    for (int i = 0; i < filtered_cloud->size(); i++) {
       PointT pt = filtered_cloud->points[i];
       if (pt.x != 0.0f && pt.y != 0.0f && pt.z != 0.0f) {
          curv_cloud->push_back(pt);
       }
    }
    
    // filter the outliers
    // this->edgeBoundaryOutlierFiltering(curv_cloud);
    
    ROS_INFO("\033[32m FILTERING OUTLIER \033[0m");

    const float search_radius_thresh = 0.01f;  // thresholds
    const int min_neigbor_thresh = 50;
    pcl::RadiusOutlierRemoval<PointT>::Ptr filter_ror(
       new pcl::RadiusOutlierRemoval<PointT>);
    filter_ror->setInputCloud(curv_cloud);
    filter_ror->setRadiusSearch(search_radius_thresh);
    filter_ror->setMinNeighborsInRadius(min_neigbor_thresh);
    filter_ror->filter(*filtered_cloud);
    
    curv_cloud->clear();
    *curv_cloud = *filtered_cloud;
    
    ROS_INFO("\033[31m COMPLETED \033[0m");


    // get interest point
    pcl::PointCloud<PointT>::Ptr anchor_points(new pcl::PointCloud<PointT>);
    this->estimateAnchorPoints(
       anchor_points, curv_cloud, curv_cloud, header);
    sensor_msgs::PointCloud2 ros_ap;
    pcl::toROSMsg(*anchor_points, ros_ap);
    ros_ap.header = header;
    this->pub_prob_.publish(ros_ap);


    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*curv_cloud, ros_cloud);
    ros_cloud.header = header;
    this->pub_pt_map_.publish(ros_cloud);
}

bool InteractiveSegmentation::estimateAnchorPoints(
    pcl::PointCloud<PointT>::Ptr anchor_points,
    const pcl::PointCloud<PointT>::Ptr convex_points,
    const pcl::PointCloud<PointT>::Ptr concave_points,
    const std_msgs::Header header) {
    if (convex_points->empty()) {
       ROS_ERROR("NO BOUNDARY REGION TO SELECT POINTS");
       return false;
    }
    // NOTE: if not concave pt then r is smallest y to selected point

    std::vector<pcl::PointIndices> cluster_convx;
    std::vector<pcl::PointIndices> cluster_concv;
    pcl::PointIndices::Ptr prob_indices(new pcl::PointIndices);
    this->doEuclideanClustering(
       cluster_convx, convex_points, prob_indices);
    if (cluster_convx.empty()) {
       return false;
    }

    // select highest point in y - dir
    float height = -FLT_MAX;
    pcl::ExtractIndices<PointT>::Ptr eifilter(new pcl::ExtractIndices<PointT>);
    eifilter->setInputCloud(convex_points);
    for (int i = 0; i < cluster_convx.size(); i++) {
       pcl::PointIndices::Ptr region_indices(new pcl::PointIndices);
       *region_indices = cluster_convx[i];
       eifilter->setIndices(region_indices);
       pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
       eifilter->filter(*tmp_cloud);
       Eigen::Vector4f center;
       pcl::compute3DCentroid<PointT, float>(*tmp_cloud, center);
       if (center(1) > height) {
          anchor_points->clear();
          pcl::copyPointCloud<PointT, PointT>(*tmp_cloud, *anchor_points);
          height = center(1);
       }
    }

    // fit plane model
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02);
    seg.setInputCloud(anchor_points);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0) {
       ROS_ERROR ("Could not estimate a planar model for the given dataset.");
       return -1;
    }
    std::cerr << "Model coefficients: " << coefficients->values[0] << " "
              << coefficients->values[1] << " "
              << coefficients->values[2] << " "
              << coefficients->values[3] << std::endl;

    float a = coefficients->values[0];
    float b = coefficients->values[1];
    float c = coefficients->values[2];
    float d = coefficients->values[3];
    /*
    Eigen::Vector3f pt_1 = Eigen::Vector3f(-d/a, 0, 0);
    Eigen::Vector3f pt_2 = Eigen::Vector3f(0, -d/b, 0);
    Eigen::Vector3f pt_3 = Eigen::Vector3f(0, 0, 0);

    Eigen::Vector3f ppt_2 = pt_2 - pt_1;
    Eigen::Vector3f ppt_3 = pt_3 - pt_1;

    anchor_points->clear();
    for (float y = 0.0f; y < 1.0f; y += 0.01f) {
       for (float x = 0.0f; x < 1.0f; x += 0.01f) {
          PointT pt;
          pt.x = pt_1(0) + ppt_2(0) * x + ppt_3(0) * y;
          pt.y = pt_1(1) + ppt_2(1) * x + ppt_3(1) * y;
          pt.z = pt_1(2) + ppt_2(2) * x + ppt_3(2) * y;
          pt.g = 255;
          anchor_points->push_back(pt);
       }
    }
    */
    
    // select a point on convex pts
    // for (int i = 0; i < convex_points->size(); i++) {
       
    // }

}

void InteractiveSegmentation::doEuclideanClustering(
    std::vector<pcl::PointIndices> &cluster_indices,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointIndices::Ptr prob_indices, const float tolerance_thresh,
    const int min_size_thresh, const int max_size_thresh) {
    cluster_indices.clear();
    if (cloud->empty()) {
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
    euclidean_clustering.extract(cluster_indices);
}

/**
 * un-used function for serivce call
 */
void InteractiveSegmentation::edgeBoundaryOutlierFiltering(
    const pcl::PointCloud<PointT>::Ptr cloud) {
    if (cloud->empty()) {
       ROS_WARN("SKIPPING OUTLIER FILTERING");
       return;
    }

    std::cout << "INPUT SIZE: " << cloud->size() << std::endl;
    
    int min_samples = 8;
    float max_distance = 0.01f;
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    interactive_segmentation::OutlierFiltering of_srv;
    of_srv.request.max_distance = static_cast<float>(max_distance);
    of_srv.request.min_samples = static_cast<int>(min_samples);
    of_srv.request.points = ros_cloud;
    if (this->srv_client_.call(of_srv)) {
       int max_label = of_srv.response.argmax_label;
       if (max_label == -1) {
          return;
       }
       std::vector<pcl::PointCloud<PointT>::Ptr> boundary_clusters;
       for (int i = 0; i < of_srv.response.labels.size(); i++) {
          if (of_srv.response.indices[i] == max_label) {
             
          }
       }
    } else {
       ROS_ERROR("ERROR! FAILED TO CALL CLUSTERING MODULE\n");
       return;
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "interactive_segmentation");
    InteractiveSegmentation is;
    ros::spin();
    return 0;
}
