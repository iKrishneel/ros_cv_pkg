
#include <interactive_segmentation/interactive_segmentation.h>
#include <vector>

InteractiveSegmentation::InteractiveSegmentation():
    min_cluster_size_(50), is_init_(false),
    num_threads_(8) {
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
    
    
    this->pub_pt_map_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
        "/interactive_segmentation/output/point_map", 1);
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/interactive_segmentation/output/image", 1);
}

void InteractiveSegmentation::subscribe() {

       this->sub_screen_pt_ = this->pnh_.subscribe(
      "input_screen", 1, &InteractiveSegmentation::screenPointCallback, this);
  
       this->sub_image_.subscribe(this->pnh_, "input_image", 1);
       this->sub_normal_.subscribe(this->pnh_, "input_normal", 1);
       this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
       this->sync_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy> >(100);
       sync_->connectInput(sub_image_, sub_normal_, sub_cloud_);
       sync_->registerCallback(boost::bind(&InteractiveSegmentation::callback,
                                           this, _1, _2, _3));
}

void InteractiveSegmentation::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_normal_.unsubscribe();
    this->sub_image_.unsubscribe();
}

void InteractiveSegmentation::screenPointCallback(
    const geometry_msgs::PointStamped &screen_msg) {
    int x = screen_msg.point.x;
    int y = screen_msg.point.y;
    this->screen_pt_ = cv::Point2i(x, y);
    this->is_init_ = true;
}

void InteractiveSegmentation::callback(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const sensor_msgs::PointCloud2::ConstPtr &normal_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    boost::mutex::scoped_lock lock(this->mutex_);
    cv::Mat image = cv_bridge::toCvShare(
       image_msg, image_msg->encoding)->image;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    std::cout << "MAIN RUNNING: " << is_init_  << "\n";
    
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
    double closest_surfel = FLT_MAX;
    uint32_t closest_surfel_index = INT_MAX;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr centroid_cloud(
       new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::Normal>::Ptr surfel_normals(
       new pcl::PointCloud<pcl::Normal>);
    int index_pos;
    for (std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr >::iterator it =
            supervoxel_clusters.begin(); it != supervoxel_clusters.end();
         it++) {
       index_pos = screen_pt_.x + (screen_pt_.y * image.cols);
       Eigen::Vector4f selected_pt = cloud->points[
          index_pos].getVector4fMap();
       Eigen::Vector4f surfel_pt = supervoxel_clusters.at(
          it->first)->centroid_.getVector4fMap();
       double dist = pcl::distances::l2(selected_pt, surfel_pt);
       if (!isnan(dist) && dist < closest_surfel) {
          closest_surfel = dist;
          closest_surfel_index = static_cast<int>(it->first);
       }
       centroid_cloud->push_back(supervoxel_clusters.at(
                                    it->first)->centroid_);
       *surfel_normals += *(supervoxel_clusters.at(it->first)->normals_);
    }
    if (closest_surfel_index == INT_MAX || isnan(closest_surfel_index)) {
       ROS_ERROR("NO SURFEL MARKED");
       return;
    }
    
    std::cout << "\033[34m 1) SELECTED \033[0m" << std::endl;

    pcl::PointCloud<PointT>::Ptr object_points(new pcl::PointCloud<PointT>);
    bool is_point_level = true;
    if (is_point_level && !supervoxel_clusters.empty()) {
       pcl::PointIndices sample_point_indices;
       // index of the sampled surfel points
       int sample_index = 0;
       sample_point_indices.indices.push_back(sample_index);
      if (is_init_) {
         // search 4 neigbours of selected surfel
         int cs_nearest = 1;
         std::vector<int> point_idx_search;
         std::vector<float> point_squared_distance;
         pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
         kdtree.setInputCloud(centroid_cloud);
         pcl::PointXYZRGBA centroid_pt = supervoxel_clusters.at(
            closest_surfel_index)->centroid_;
         if (isnan(centroid_pt.x) || isnan(centroid_pt.y) ||
             isnan(centroid_pt.z)) {
            return;
         }
         int search_out = kdtree.nearestKSearch(
            centroid_pt, cs_nearest, point_idx_search, point_squared_distance);
         
         // just use the origin cloud and norm
         int k = 50;
         this->estimatePointCloudNormals<int>(cloud, normals, k, true);
         Eigen::Vector4f attention_normal = this->cloudMeanNormal(
            supervoxel_clusters.at(closest_surfel_index)->normals_);
         Eigen::Vector4f attention_centroid = centroid_pt.getVector4fMap();
         
         // select few points close to centroid as true object


         cv::Mat weight_map;
         this->surfelSamplePointWeightMap(cloud, normals, centroid_pt,
                                          attention_normal, weight_map);
         
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

            /*
            cv::Mat sample_weight_map;
            Eigen::Vector4f idx_attn_normal = surfel_normals->points[
                idx].getNormalVector4fMap();
            this->surfelSamplePointWeightMap(cloud, normals, neigh_pt,
                                             idx_attn_normal,
                                             sample_weight_map);
            cv::Mat tmp;
            cv::add(weight_map, sample_weight_map, tmp);
            weight_map = tmp.clone();
            */
         }
         cv::normalize(weight_map, weight_map, 0, 1,
                       cv::NORM_MINMAX, -1, cv::Mat());
         
         // normalize weights **REMOVE THIS CLOUD**
         pcl::PointCloud<PointT>::Ptr weight_cloud(new pcl::PointCloud<PointT>);
         for (int x = 0; x < weight_map.rows; x++) {
            cloud->points[x].r = weight_map.at<float>(x, 0) * 255.0f;
            cloud->points[x].g = weight_map.at<float>(x, 0) * 255.0f;
            cloud->points[x].b = weight_map.at<float>(x, 0) * 255.0f;
         }
         *weight_cloud = *cloud;
         
         std::cout << cloud->size() << "\t" << normals->size() << "\t"
                   << weight_cloud->size() << "\n";
        std::cout << "\033[34m 3) COMPUTING WEIGHTS \033[0m" << std::endl;
        
        // weights for graph cut
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
       
        // select the object mask
        /*
        cv::Mat object_mask;
        cv::Rect rect = cv::Rect(0, 0, 0, 0);
        this->attentionSurfelRegionMask(conv_weights, screen_pt_,
                                        object_mask, rect);
        cv::cvtColor(image, image, CV_RGB2BGR);
        this->graphCutSegmentation(image, object_mask, rect, 1);

        cv::imshow("weights", conv_weights);
        cv::waitKey(3);
        */
        
        // get indices of the probable object mask
        pcl::PointIndices::Ptr prob_object_indices(new pcl::PointIndices);
        pcl::PointCloud<PointT>::Ptr prob_object_cloud(
           new pcl::PointCloud<PointT>);
        this->attentionSurfelRegionPointCloudMask(cloud, attention_centroid,
                                                  cloud_msg->header,
                                                  prob_object_cloud,
                                                  prob_object_indices);

        
        // TMP
        pcl::PointCloud<PointT>::Ptr cov_cloud(new pcl::PointCloud<PointT>);
        this->computePointCloudCovarianceMatrix(prob_object_cloud, cov_cloud);
        cloud->clear();
        *cloud = *cov_cloud;
        
        
        // this->organizedMinCutMaxFlowSegmentation(
        //    prob_object_cloud, index_pos);
        
        
        /*
        // segmentation
        pcl::PointCloud<PointT>::Ptr object_cloud(new pcl::PointCloud<PointT>);
        this->objectMinCutSegmentation(cloud, object_points,
                                       prob_object_cloud, object_cloud);
        cloud->clear();
        *cloud = *object_cloud;
        */
        
        std::cout << cloud->size() << "\t" << normals->size() << std::endl;
      }   // end if
      
      // cv::Mat saliency_img;
      // this->generateFeatureSaliencyMap(image, saliency_img);
      cv_bridge::CvImage pub_img(
          image_msg->header, sensor_msgs::image_encodings::BGR8, image);
      this->pub_image_.publish(pub_img.toImageMsg());
    }  // end if
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}


void InteractiveSegmentation::surfelSamplePointWeightMap(
     const pcl::PointCloud<PointT>::Ptr cloud,
     const pcl::PointCloud<pcl::Normal>::Ptr normals,
     const pcl::PointXYZRGBA &centroid_pt,
     const Eigen::Vector4f attention_normal,
     cv::Mat &weights
     /*pcl::PointCloud<PointT>::Ptr weights*/) {
     if (cloud->empty() || normals->empty()) {
       return;
     }
     Eigen::Vector4f attention_centroid = centroid_pt.getVector4fMap();
     cv::Mat connectivity_weights;
     cv::Mat orientation_weights;
     for (int i = 0; i < normals->size(); i++) {
       Eigen::Vector4f current_pt = cloud->points[i].getVector4fMap();
       Eigen::Vector4f d = (attention_centroid - current_pt) /
           (attention_centroid - current_pt).norm();
       Eigen::Vector4f current_normal =
           normals->points[i].getNormalVector4fMap();
       // float connection = (attention_normal - current_normal).dot(d);
       float connection = (current_pt - attention_centroid).dot(
           current_normal);
       if (connection <= 0.0f || isnan(connection)) {
         connection = 0.0f;
       } else {
         connection = cos(current_normal.dot(attention_normal))/
             (1.0f * M_PI);
       }
       connectivity_weights.push_back(connection);

       // orientation with respect to marked
       Eigen::Vector3f viewPointVec = (cloud->points[i].getVector3fMap() -
                                       centroid_pt.getVector3fMap());
       Eigen::Vector3f surfaceNormalVec = normals->points[
           i].getNormalVector3fMap() - attention_normal.head<3>();
       float cross_norm = static_cast<float>(
           surfaceNormalVec.cross(viewPointVec).norm());
       float scalar_prod = static_cast<float>(
           surfaceNormalVec.dot(viewPointVec));
       float angle = atan2(cross_norm, scalar_prod);
       float view_pt_weight = angle/(2.0 * CV_PI);
       view_pt_weight = std::exp(-2.0f * view_pt_weight);
       view_pt_weight * this->whiteNoiseKernel(view_pt_weight);
       orientation_weights.push_back(view_pt_weight);
     }
     cv::normalize(connectivity_weights, connectivity_weights, 0, 1,
                   cv::NORM_MINMAX, -1, cv::Mat());
     cv::normalize(orientation_weights, orientation_weights, 0, 1,
                   cv::NORM_MINMAX, -1, cv::Mat());

     // smoothing
     /*
     const int filter_lenght = 3;
     cv::GaussianBlur(connectivity_weights, connectivity_weights,
                      cv::Size(filter_lenght, filter_lenght), 0, 0);
     cv::GaussianBlur(orientation_weights, orientation_weights,
                      cv::Size(filter_lenght, filter_lenght), 0, 0);
     
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
     for (int i = 0; i < connectivity_weights.rows; i++) {
       float pix_val = connectivity_weights.at<float>(i, 0);
       connectivity_weights.at<float>(i, 0) = pix_val *
           this->whiteNoiseKernel(pix_val);
       pix_val *= this->whiteNoiseKernel(pix_val);
       pix_val *= orientation_weights.at<float>(i, 0);

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
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    for (int i = 0; i < weight_cloud->size(); i++) {
       PointT pt = weight_cloud->points[i];
       if (pt.r > 0 && pt.b > 0 && pt.g > 0 &&
           !isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z)) {
          cloud->push_back(pt);
       }
    }
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> euclidean_clustering;
    euclidean_clustering.setClusterTolerance(0.01f);
    euclidean_clustering.setMinClusterSize(10);
    euclidean_clustering.setMaxClusterSize(25000);
    euclidean_clustering.setSearchMethod(tree);
    euclidean_clustering.setInputCloud(cloud);
    // euclidean_clustering.setIndices(indices);
    euclidean_clustering.extract(cluster_indices);
    double min_distance = DBL_MAX;
    pcl::ExtractIndices<PointT>::Ptr eifilter(new pcl::ExtractIndices<PointT>);
    eifilter->setInputCloud(cloud);
    // pcl::PointCloud<PointT>::Ptr object_cloud(new pcl::PointCloud<PointT>);
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

void InteractiveSegmentation::objectMinCutSegmentation(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<PointT>::Ptr object_points,
    const pcl::PointCloud<PointT>::Ptr prob_object_cloud,
    pcl::PointCloud<PointT>::Ptr object_cloud) {
    if (cloud->empty() || object_points->empty()) {
       return;
    }

    std::cout << "\033[35mFILTER SIZE: " << prob_object_cloud->size()
              << "\t" << object_points->size() << std::endl;
    
    pcl::MinCutSegmentation<PointT> segmentation;
    segmentation.setInputCloud(prob_object_cloud);
    segmentation.setForegroundPoints(object_points);

    const float sigma = 0.0001f;
    const float radius = 0.1f;
    const int k = 10;
    const float source_weight = 0.8f;

    segmentation.setSigma(sigma);
    segmentation.setRadius(radius);
    segmentation.setNumberOfNeighbours(k);
    segmentation.setSourceWeight(source_weight);

    std::vector<pcl::PointIndices> clusters;
    segmentation.extract(clusters);
    *object_cloud = *(segmentation.getColoredCloud());
    
    std::cout << "MAX FLOW: " << segmentation.getMaxFlow()
              << "\t" << object_cloud->size() << std::endl;
}


void InteractiveSegmentation::attentionSurfelRegionMask(
    const cv::Mat conv_weights, const cv::Point2i screen_pt,
    cv::Mat &object_mask, cv::Rect &rect) {
    if (conv_weights.empty()) {
       return;
    }
    // create mask
    cv::Mat img_bw = cv::Mat::zeros(conv_weights.size(), CV_8UC1);
    for (int j = 0; j < img_bw.rows; j++) {
       for (int i = 0; i < img_bw.cols; i++) {
          if (conv_weights.at<float>(j, i) > 0.0f) {
             img_bw.at<uchar>(j, i) = 255;
          }
       }
    }
    // compute the contours
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    cv::Sobel(img_bw, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::Sobel(img_bw, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, img_bw);
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img_bw, contours, hierarchy, CV_RETR_TREE,
                     CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    // check the contour point belongs to
    cv::Mat mask_img = cv::Mat::zeros(img_bw.rows, img_bw.cols, CV_8UC3);
    int index = -1;
    for (int i = 0; i < contours.size(); i++) {
       double dist = cv::pointPolygonTest(
          contours[i], cv::Point2f(static_cast<float>(screen_pt.x),
                                   static_cast<float>(screen_pt.y)),
          false);
       if (dist > -1) {
          index = i;
          cv::drawContours(mask_img, contours, i,
                           cv::Scalar(255, 255, 255), CV_FILLED);
       }
    }
    if (index != -1) {
       rect = cv::boundingRect(contours[index]);
       const int padding = 20;
       rect.x -= padding;
       rect.y -= padding;
       rect.width += (2 * padding);
       rect.height += (2 * padding);
    }
    object_mask = cv::Mat::zeros(conv_weights.size(), CV_32F);
    for (int j = 0; j < mask_img.rows; j++) {
       for (int i = 0; i < mask_img.cols; i++) {
          cv::Vec3b pix_val = mask_img.at<cv::Vec3b>(j, i);
          if (pix_val[0] > 0) {
             float conv_weight = conv_weights.at<float>(j, i);
             object_mask.at<float>(j, i) = conv_weight;
             // object_mask.at<float>(j, i) = 1.0f;
          }
       }
    }
    // cv::bitwise_or(conv_weights, object_mask, object_mask);
    // cv::imshow("contour", mask_img);
    cv::imshow("masked", object_mask);
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
          // sum_of_eigens_ptr[i] = (eigen_vals(0) - eigen_vals(2)) / eigen_vals(1);
          
          curvature_ptr[i] = (eigen_vals(0) / sum_eigen);

          // ****** TEMP
          double dist = pcl::distances::l2(
             centroid_pt.getVector4fMap(), center);
          if (dist < distance) {
             index = i;
          }
       }
    }

    
    // pcl::PointCloud<PointT>::Ptr new_cloud(new pcl::PointCloud<PointT>);
    for (int i = 0; i < cloud->size(); i++) {
      float dist_diff_sum = std::pow(sum_of_eigens_ptr[index] - sum_of_eigens_ptr[i], 2);
      dist_diff_sum += dist_diff_sum;
      dist_diff_sum = std::sqrt(dist_diff_sum);
      // std::cout << "DIST: " << dist_diff_sum  << "\t";
       
      float dist_diff_cur = std::pow(curvature_ptr[index] - curvature_ptr[i], 2);
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


void InteractiveSegmentation::viewPointSurfaceNormalOrientation(
    pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr cloud_normal,
    const Eigen::Vector3f cenVec, const Eigen::Vector3f norVec) {
    if (cloud->empty() || cloud_normal->empty()) {
      ROS_ERROR("ERROR: Point Cloud | Normal vector is empty...");
      return;
    }
    pcl::PointCloud<PointT>::Ptr point_orientation(
       new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(*cloud, *point_orientation);
#ifdef _OPENMP
#pragma omp parallel for shared(point_orientation) \
   num_threads(this->num_threads_)
#endif
    for (int i = 0; i < cloud->size(); i++) {
       Eigen::Vector3f viewPointVec =
          (cloud->points[i].getVector3fMap() - cenVec);
       Eigen::Vector3f surfaceNormalVec = cloud_normal->points[
          i].getNormalVector3fMap() - norVec;
      float cross_norm = static_cast<float>(
          surfaceNormalVec.cross(viewPointVec).norm());
      float scalar_prod = static_cast<float>(
          surfaceNormalVec.dot(viewPointVec));
      float angle = atan2(cross_norm, scalar_prod);
      float pix_val = angle/(2.0 * CV_PI);
      pix_val = std::exp(-2.0f * pix_val);
      pix_val * this->whiteNoiseKernel(pix_val);
        
      PointT *pt = &point_orientation->points[i];
      pt->x = cloud->points[i].x;
      pt->y = cloud->points[i].y;
      pt->z = cloud->points[i].z;
      pt->r = pix_val * 255;
      pt->g = pix_val * 255;
      pt->b = pix_val * 255;
    }
    cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*point_orientation, *cloud);
}

void InteractiveSegmentation::normalNeigbourOrientation(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    pcl::PointCloud<PointT>::Ptr orientation, const int k) {
    if (cloud->size() != normals->size()) {
       ROS_ERROR("ERROR. INCORRECT SIZE");
      return;
    }
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);
    pcl::copyPointCloud<PointT, PointT>(*cloud, *orientation);
#ifdef _OPENMP
#pragma omp parallel for shared(orientation) num_threads(this->num_threads_)
#endif
    for (int i = 0; i < cloud->size(); i++) {
       std::vector<int> point_idx_search;
       std::vector<float> point_squared_distance;
       int search_out = 0;
       bool is_knn = true;       
       if (is_knn) {
          search_out = kdtree.nearestKSearch(
             cloud->points[i], 40, point_idx_search, point_squared_distance);
       } else {
          search_out = kdtree.radiusSearch(
             cloud->points[i], 0.02, point_idx_search, point_squared_distance);
       }
      
       Eigen::Vector4f normal = normals->points[i].getNormalVector4fMap();
       float sum = 0.0f;
       for (int j = 0; j < point_idx_search.size(); j++) {
          int index = point_idx_search.at(j);
          Eigen::Vector4f n_normal =
             normals->points[index].getNormalVector4fMap();
          sum += normal.dot(n_normal);
          float diff = (cloud->points[i].getVector4fMap() -
                        cloud->points[index].getVector4fMap()).dot(normal);
          if (diff > 0.0f) {
             // sum += static_cast<float>(
             // std::pow(1 - (normal.dot(n_normal)), 2));
          } else {
             // sum = static_cast<float>(1 - (normal.dot(n_normal)));
          }
       }
       sum /= static_cast<float>(point_idx_search.size());
       PointT pt = cloud->points[i];
       // pt.r = sum * 255;
       // pt.g = sum * 255;
       // pt.b = sum * 255;
       // pt.r = normals->points[i].normal_x * 1.0;
       // pt.g = normals->points[i].normal_y * 1.0;
       // pt.b = normals->points[i].normal_z * 1.0;
       // std::cout << pt.r + pt.g + pt.b << std::endl;
       //-----
       // Eigen::Vector4f normal = normals->points[i].getNormalVector4fMap();
       Eigen::Vector4f gravity = Eigen::Vector4f(0, 1.0, 0.0, 1.0);
       // float sum = (normal-gravity).dot(cloud->points[i].getVector4fMap());
       // // float sum = (normal).dot(gravity) *
       // // float sum =   (cloud->points[i].getVector4fMap()).dot(gravity);
       // PointT pt = cloud->points[i];
       pt.r = sum * 255;
       pt.g = sum * 255;
       pt.b = sum * 255;
       orientation->points[i] = pt;
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


void InteractiveSegmentation::pointLevelSimilarity(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    const std_msgs::Header header) {
    if (cloud->empty() || normals->empty()) {
       return;
    }
     pcl::PointCloud<PointT>::Ptr out_cloud(new pcl::PointCloud<PointT>);
     pcl::copyPointCloud<PointT, PointT>(*cloud, *out_cloud);
     
     pcl::KdTreeFLANN<PointT> kdtree;
     kdtree.setInputCloud(cloud);
     bool is_knn = true;
     float search_dim = 0.025f;
#ifdef _OPENMP
     #pragma omp parallel for
#endif
     for (int i = 0; i < cloud->size(); i++) {

        pcl::PointXYZHSV hsv;
       pcl::PointXYZRGBtoXYZHSV(cloud->points[i], hsv);
       std::vector<int> point_idx_search;
       std::vector<float> point_squared_distance;
       PointT pt = cloud->points[i];
       int search_out = 0;
       if (is_knn) {
         search_out = kdtree.nearestKSearch(
             pt, 40, point_idx_search, point_squared_distance);
       } else {
         search_out = kdtree.radiusSearch(
             pt, search_dim, point_idx_search, point_squared_distance);
       }
       double sum = 0.0;
       for (size_t k = 0; k < point_idx_search.size(); k++) {
         int index = point_idx_search[k];

         pcl::PointXYZHSV n_hsv;
         pcl::PointXYZRGBtoXYZHSV(cloud->points[index], n_hsv);

         double dist_color = std::sqrt(std::pow((hsv.h - n_hsv.h), 2) +
                                       std::pow((hsv.s - n_hsv.s), 2));
         dist_color = (255.0 - dist_color)/255.0;
         
         Eigen::Vector4f i_point = cloud->points[i].getVector4fMap();
         Eigen::Vector4f k_point = cloud->points[k].getVector4fMap();
         double dist_point = pcl::distances::l2(i_point, k_point);
         
         double dist_fpfh = 0.0;
         // dist_fpfh = cv::compareHist(fpfh_hist.row(i),
         //                                    fpfh_hist.row(index),
         //                                    CV_COMP_CORREL);

         Eigen::Vector4f norm = Eigen::Vector4f(
            normals->points[i].normal_x,
            normals->points[i].normal_y,
            normals->points[i].normal_z, 1.0f);
         Eigen::Vector4f n_norm = Eigen::Vector4f(
            normals->points[index].normal_x,
            normals->points[index].normal_y,
            normals->points[index].normal_z, 1.0f);
         dist_fpfh = pcl::distances::l2(norm, n_norm);

         double distance = std::sqrt(
            std::pow(dist_color, 2)
            // + std::pow(dist_fpfh, 2)
            + std::pow(dist_point, 2)
            );
         
         sum += distance;
       }
       sum /= static_cast<double>(point_idx_search.size());
       
       double intensity = 255.0;
       out_cloud->points[i].r = intensity * sum;
       out_cloud->points[i].b = intensity * sum;
       out_cloud->points[i].g = intensity * sum;
     }

     sensor_msgs::PointCloud2 ros_cloud;
     pcl::toROSMsg(*out_cloud, ros_cloud);
     ros_cloud.header = header;
     this->pub_pt_map_.publish(ros_cloud);

  }


void InteractiveSegmentation::computePointFPFH(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    cv::Mat &histogram) const {
    if (cloud->empty() || normals->empty()) {
      ROS_ERROR("-- ERROR: cannot compute FPFH");
      return;
    }
    pcl::FPFHEstimationOMP<PointT, pcl::Normal, FPFHS> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(0.05);
    pcl::PointCloud<FPFHS>::Ptr fpfhs(new pcl::PointCloud<FPFHS>());
    fpfh.compute(*fpfhs);
    const int hist_dim = 33;
    histogram = cv::Mat::zeros(
        static_cast<int>(fpfhs->size()), hist_dim, CV_32F);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(histogram)
#endif
    for (int i = 0; i < fpfhs->size(); i++) {
      for (int j = 0; j < hist_dim; j++) {
        histogram.at<float>(i, j) = fpfhs->points[i].histogram[j];
      }
    }
    cv::normalize(histogram, histogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
}

bool InteractiveSegmentation::localVoxelConvexityCriteria(
    Eigen::Vector4f c_centroid, Eigen::Vector4f c_normal,
    Eigen::Vector4f n_centroid, Eigen::Vector4f n_normal,
    const float threshold) {
    c_centroid(3) = 0.0f;
    c_normal(3) = 0.0f;
    if ((n_centroid - c_centroid).dot(n_normal) > 0) {
        return true;
    } else {
        return false;
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
    ne.setNumberOfThreads(16);
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
    const float pix_val) {
    float weight = 1.0f / (std::sqrt(2.0f * M_PI) *
                           exp(-1.0f *  (pix_val)));
    return weight;
}

void InteractiveSegmentation::organizedMinCutMaxFlowSegmentation(
    pcl::PointCloud<PointT>::Ptr cloud, const int selected_index) {
    if (cloud->empty() || cloud->height == 1) {
       ROS_ERROR("ERROR: Empty cloud for segmentation");
       return;
    }

    // make the neigbourbood
    int knearest = 8;
    std::vector<int> point_idx_search;
    std::vector<float> point_squared_distance;
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);
    for (int i = 0; i < cloud->size(); i++) {
       PointT centroid_pt = cloud->points[i];
       if (!isnan(centroid_pt.x) || !isnan(centroid_pt.y) ||
           !isnan(centroid_pt.z)) {
          int search_out = kdtree.nearestKSearch(
             centroid_pt, knearest, point_idx_search, point_squared_distance);
       }
    }
    
    const unsigned int D = 2;
    typedef boost::grid_graph<D> Graph;
    typedef boost::graph_traits<Graph>::vertex_descriptor VertexDescriptor;
    typedef boost::graph_traits<Graph>::edge_descriptor EdgeDescriptor;
    typedef boost::graph_traits<Graph>::vertices_size_type VertexIndex;
    typedef boost::graph_traits<Graph>::edges_size_type EdgeIndex;

    boost::array<std::size_t, D> lengths = {{cloud->width, cloud->height}};
    Graph graph(lengths, false);

    const int kSize = cloud->size();
    float weights[kSize];
    for (int i = 0; i < cloud->size(); i++) {
       weights[i] = static_cast<float>(cloud->points[i].r);
    }
    std::vector<int> groups(num_vertices(graph));
    std::vector<float> residual_capacity(num_edges(graph));
    std::vector<float> capacity(num_edges(graph));
    std::vector<EdgeDescriptor> reverse_edges(num_edges(graph));

    BGL_FORALL_EDGES(e, graph, Graph) {
        VertexDescriptor src = source(e, graph);
        VertexDescriptor tgt = target(e, graph);
        VertexIndex source_idx = get(boost::vertex_index, graph, src);
        VertexIndex target_idx = get(boost::vertex_index, graph, tgt);
        EdgeIndex edge_idx = get(boost::edge_index, graph, e);
        capacity[edge_idx] = fabs(weights[source_idx] - weights[target_idx]);
        reverse_edges[edge_idx] = edge(tgt, src, graph).first;
    }
    int sink_idx = selected_index;
    int target_idx = 0;
    VertexDescriptor s = vertex(sink_idx, graph);
    VertexDescriptor t = vertex(target_idx, graph);
    
     boykov_kolmogorov_max_flow(
        graph,
        make_iterator_property_map(&capacity[0], get(
                                      boost::edge_index, graph)),
        make_iterator_property_map(&residual_capacity[0],
                                   get(boost::edge_index, graph)),
        make_iterator_property_map(&reverse_edges[0],
                                   get(boost::edge_index, graph)),
        make_iterator_property_map(&groups[0], get(boost::vertex_index, graph)),
        get(boost::vertex_index, graph), s, t);

     for (size_t index=0; index < groups.size(); ++index) {
        if ((index%lengths[0] == 0) && index) {
           std::cout << std::endl;
        }
        std::cout << groups[index] << " ";
     }
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "interactive_segmentation");
    InteractiveSegmentation is;
    ros::spin();
    return 0;
}
