
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
    
    // std::vector<int> index;
    // pcl::removeNaNFromPointCloud<PointT>(*cloud, *cloud, index);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    // int neigbour_size = 10;
    // this->estimatePointCloudNormals<int>(
    //    cloud, normals, neigbour_size, true);
    
    // this->pointLevelSimilarity(cloud, normals, cloud_msg->header);
    
    
    // this->pointCloudEdge(cloud, image, 10);

    bool is_surfel_level = false;
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
    if (is_surfel_level) {
       this->surfelLevelObjectHypothesis(cloud, normals, supervoxel_clusters);
       sensor_msgs::PointCloud2 ros_voxels;
       jsk_recognition_msgs::ClusterPointIndices ros_indices;
       this->publishSupervoxel(supervoxel_clusters,
                               ros_voxels, ros_indices, cloud_msg->header);
       this->pub_voxels_.publish(ros_voxels);
       this->pub_indices_.publish(ros_indices);
    }

    bool is_point_level = true;
    if (is_point_level /*&& !supervoxel_clusters.empty()*/) {

      pcl::KdTreeFLANN<PointT> kdtree;
      kdtree.setInputCloud(cloud);
      // for (std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr >::iterator it =
      // supervoxel_clusters.begin(); it != supervoxel_clusters.end(); it++) {
      // get the attention points
      // pcl::PointXYZRGBA centroid_pt = supervoxel_clusters.at(
      //     it->first)->centroid_;
      if (is_init_) {
        int index_pos = screen_pt_.x + (screen_pt_.y * image.cols);
        PointT centroid_pt = cloud->points[index_pos];
        PointT attention_pt;
        attention_pt.x = centroid_pt.x;
        attention_pt.y = centroid_pt.y;
        attention_pt.z = centroid_pt.z;
        attention_pt.r = centroid_pt.r;
        attention_pt.g = centroid_pt.b;
        attention_pt.b = centroid_pt.g;

        if (isnan(attention_pt.x) || isnan(attention_pt.y) ||
            isnan(attention_pt.z)) {
          return;
        }
        
        // compute point cloud normals
        // std::vector<int> index;
        // pcl::removeNaNFromPointCloud<PointT>(*cloud, *cloud, index);
        float k = 0.05f;
        this->estimatePointCloudNormals<float>(cloud, normals, k, false);

        // find some neigbours
        std::vector<int> point_idx_search;
        std::vector<float> point_squared_distance;
        int search_out = kdtree.nearestKSearch(
            attention_pt, 10, point_idx_search, point_squared_distance);

        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        int icounter = 0;
        for (int i = 0; i < point_idx_search.size(); i++) {
          int index = point_idx_search.at(i);
          pcl::Normal normal = normals->points[index];
          if (!isnan(normal.normal_x) || !isnan(normal.normal_y) ||
              !isnan(normal.normal_z)) {
            x += normal.normal_x;
            y += normal.normal_y;
            z += normal.normal_z;
            icounter++;
          }
        }
        Eigen::Vector4f attention_normal = Eigen::Vector4f(
            x/static_cast<float>(icounter),
            y/static_cast<float>(icounter),
            z/static_cast<float>(icounter), 1.0f);
        
        std::cout << cloud->size() << "\t" << normals->size()  << "\n";
        Eigen::Vector4f attention_centroid = centroid_pt.getVector4fMap();
        for (int i = 0; i < normals->size(); i++) {
          if (i != index_pos) {
            Eigen::Vector4f current_pt = cloud->points[i].getVector4fMap();
            Eigen::Vector4f d = (attention_centroid - current_pt) /
                (attention_centroid - current_pt).norm();
            Eigen::Vector4f current_normal =
               normals->points[i].getNormalVector4fMap();
            float connection = (attention_normal - current_normal).dot(d);
            // float connection = (current_normal).dot(d);
            
            if (connection <= this->convex_threshold_ || isnan(connection)
                /*||(acos(attention_normal.dot(current_normal)) < M_PI/18)*/) {
              cloud->points[i].r = 255;
              cloud->points[i].b = 0;
              cloud->points[i].g = 0;
            } else {
               cloud->points[i].r = 255 * connection;
               cloud->points[i].b = 255 * connection;
               cloud->points[i].g = 255 * connection;
            }
          }
        }
        */
        
        // this->viewPointSurfaceNormalOrientation(
        //    cloud, normals, centroid_pt.getVector3fMap(),
        //    normals->points[index_pos].getNormalVector3fMap());
        
        
        std::cout << cloud->size() << "\t" << normals->size() << std::endl;
        // pcl::PointCloud<PointT>::Ptr orientation(new pcl::PointCloud<PointT>);
        // this->normalNeigbourOrientation(cloud, normals, orientation, k);
        // cloud->clear();
        // *cloud = *orientation;
      }
      

      

      
      cv::Mat saliency_img;
      this->generateFeatureSaliencyMap(image, saliency_img);
      cv_bridge::CvImage pub_img(
          image_msg->header, sensor_msgs::image_encodings::BGR8, saliency_img);
      this->pub_image_.publish(pub_img.toImageMsg());
    }
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
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
    cloud->clear();
    for (std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>::iterator it =
            supervoxel_clusters.begin(); it != supervoxel_clusters.end();
         it++) {
       voxel_labels[it->first] = -1;
       pcl::Supervoxel<PointT>::Ptr supervoxel =
          supervoxel_clusters.at(it->first);
       *normals = *normals + *(supervoxel->normals_);
       *cloud = *cloud + *(supervoxel->voxels_);
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
       
       // Eigen::Vector3f viewPointVec =
       //    Eigen::Vector3f(0, 1.0, 0);

       Eigen::Vector3f surfaceNormalVec = cloud_normal->points[
          i].getNormalVector3fMap() - norVec;
      float cross_norm = static_cast<float>(
          surfaceNormalVec.cross(viewPointVec).norm());
      float scalar_prod = static_cast<float>(
          surfaceNormalVec.dot(viewPointVec));
      float angle = atan2(cross_norm, scalar_prod);
      
      // std::cout << angle << std::endl;
      
      // if (angle * (180/CV_PI) >= 0 && angle * (180/CV_PI) <= 180) {
        // cv::Scalar jmap = JetColour(angle/(CV_PI), 0, 1);
        float pix_val = 1.0f - angle/(2.0 * CV_PI);
        PointT *pt = &point_orientation->points[i];
        pt->x = cloud->points[i].x;
        pt->y = cloud->points[i].y;
        pt->z = cloud->points[i].z;
        pt->r = pix_val * 255;
        pt->g = pix_val * 255;
        pt->b = pix_val * 255;
        // }
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




void InteractiveSegmentation::pointLevelSimilarity(
     const pcl::PointCloud<PointT>::Ptr cloud,
     const pcl::PointCloud<pcl::Normal>::Ptr normals,
     const std_msgs::Header header) {
     if (cloud->empty() || normals->empty()) {
        return;
     }

     // get the features
     // pcl::PointCloud<PointT>::Ptr dist_map(new pcl::PointCloud<PointT>);
     // cv::Mat fpfh_hist;
     // this->computePointFPFH(cloud, normals, fpfh_hist);
     
     // compute weight
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

void InteractiveSegmentation::pointCloudEdge(
    pcl::PointCloud<PointT>::Ptr cloud,
    const cv::Mat &image, const int contour_thresh) {
    if (image.empty()) {
       ROS_ERROR("-- Cannot eompute edge of empty image");
       return;
    }
    cv::Mat edge_img;
    cv::GaussianBlur(image, edge_img, cv::Size(3, 3), 1);
    cv::Canny(edge_img, edge_img, 50, 150, 3, true);
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat cont_img = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::findContours(edge_img, contours, hierarchy, CV_RETR_LIST,
                     CV_CHAIN_APPROX_NONE, cv::Point(0, 0));
    std::vector<std::vector<cv::Point> > selected_contours;
    for (int i = 0; i < contours.size(); i++) {
       if (cv::contourArea(contours[i]) > contour_thresh) {
          selected_contours.push_back(contours[i]);
          cv::drawContours(cont_img, contours, i, cv::Scalar(0, 255, 0), 2);
          for (int j = 0; j < contours[i].size(); j++) {
             cv::circle(cont_img, contours[i][j], 1,
                        cv::Scalar(255, 0, 0), -1);
          }
       }
    }
    std::vector<std::vector<EdgeNormalDirectionPoint> > normal_points;
    std::vector<std::vector<cv::Point> > tangents;
    this->computeEdgeCurvature(
       cont_img, selected_contours, tangents, normal_points);
    
    cv::imshow("Contours", cont_img);
    cv::waitKey(3);
    
    // pcl::PointCloud<pcl::Normal>::Ptr normals(
    //    new pcl::PointCloud<pcl::Normal>);
    // this->estimatePointCloudNormals(cloud, normals, 0.03f, false);
    pcl::PointCloud<PointT>::Ptr concave_cloud(new pcl::PointCloud<PointT>);
    for (int j = 0; j < normal_points.size(); j++) {
       for (int i = 0; i < normal_points[j].size(); i++) {
          EdgeNormalDirectionPoint point_info = normal_points[j][i];
          cv::Point2f n_pt1 = point_info.normal_pt1;
          cv::Point2f n_pt2 = point_info.normal_pt2;
          cv::Point2f e_pt = (n_pt1 + n_pt2);
          e_pt = cv::Point2f(e_pt.x/2, e_pt.y/2);
          int ept_index = e_pt.x + (e_pt.y * image.cols);
          int pt1_index = n_pt1.x + (n_pt1.y * image.cols);
          int pt2_index = n_pt2.x + (n_pt2.y * image.cols);
          
          if (pt1_index > -1 && pt2_index > -1 &&  ept_index > -1 &&
              pt1_index < static_cast<int>(cloud->size() + 1) &&
              pt2_index < static_cast<int>(cloud->size() + 1) &&
              ept_index < static_cast<int>(cloud->size() + 1)) {
             Eigen::Vector3f ne_pt1 = cloud->points[pt1_index].getVector3fMap();
             Eigen::Vector3f ne_pt2 = cloud->points[pt2_index].getVector3fMap();
             Eigen::Vector3f ne_cntr = ((ne_pt1 - ne_pt2) / 2) + ne_pt2;
             Eigen::Vector3f e_pt = cloud->points[ept_index].getVector3fMap();

             
             PointT pt = cloud->points[ept_index];
             if (ne_cntr(2) < e_pt(2)
                 /*|| isnan(e_pt(2)) || isnan(ne_cntr(2))*/) {
                pt.r = 0;
                pt.b = 0;
                pt.g = 255;
                concave_cloud->push_back(pt);
             }             
             /*
             pcl::Normal n1 = normals->points[pt1_index];
             pcl::Normal n2 = normals->points[pt2_index];
             Eigen::Vector3f n1_vec = Eigen::Vector3f(
                n1.normal_x, n1.normal_y, n1.normal_z);
             Eigen::Vector3f n2_vec = Eigen::Vector3f(
                n2.normal_x, n2.normal_y, n2.normal_z);
             float cos_theta = n1_vec.dot(n2_vec) /
                (n1_vec.norm() * n2_vec.norm());
             float angle = std::acos(cos_theta);
             std::cout << "Angle: " << angle * 180.0f/CV_PI << std::endl;
             if (angle < CV_PI/3 && !isnan(angle)) {
                PointT pt = cloud->points[ept_index];
                pt.r = 255;
                pt.b = 0;
                pt.g = 0;
                concave_cloud->push_back(pt);
             }
             */
          }
       }
    }
    cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*concave_cloud, *cloud);
}

void InteractiveSegmentation::computeEdgeCurvature(
    const cv::Mat &image,
    const std::vector<std::vector<cv::Point> > &contours,
    std::vector<std::vector<cv::Point> > &tangents,
    std::vector<std::vector<EdgeNormalDirectionPoint> > &normal_points) {
    if (contours.empty()) {
       ROS_ERROR("-- no contours found");
       return;
    }
    normal_points.clear();
    cv::Mat img = image.clone();
    for (int j = 0; j < contours.size(); j++) {
       std::vector<cv::Point> tangent;
       std::vector<float> t_gradient;
       std::vector<EdgeNormalDirectionPoint> norm_tangt;
       cv::Point2f edge_pt = contours[j].front();
       cv::Point2f edge_tngt = contours[j].back() - contours[j][1];
       tangent.push_back(edge_tngt);
       float grad = (edge_tngt.y - edge_pt.y) / (edge_tngt.x - edge_pt.x);
       t_gradient.push_back(grad);
       const int neighbor_pts = 0;
       if (contours[j].size() > 2) {
          for (int i = sizeof(char) + neighbor_pts;
               i < contours[j].size() - sizeof(char) - neighbor_pts;
               i++) {
             edge_pt = contours[j][i];
             edge_tngt = contours[j][i-1-neighbor_pts] -
                contours[j][i+1+neighbor_pts];
            tangent.push_back(edge_tngt);
            grad = (edge_tngt.y - edge_pt.y) / (edge_tngt.x - edge_pt.x);
            t_gradient.push_back(grad);
            cv::Point2f pt1 = edge_tngt + edge_pt;
            cv::Point2f trans = pt1 - edge_pt;
            cv::Point2f ortho_pt1 = edge_pt + cv::Point2f(-trans.y, trans.x);
            cv::Point2f ortho_pt2 = edge_pt - cv::Point2f(-trans.y, trans.x);

            float theta = std::atan2(ortho_pt1.y - ortho_pt2.y ,
                                     ortho_pt1.x - ortho_pt2.x);
            const float lenght = 10.0f;
            float y1 = std::sin(theta) * lenght;
            float x1 = std::cos(theta) * lenght;
            float y2 = std::sin(CV_PI + theta) * lenght;
            float x2 = std::cos(CV_PI + theta) * lenght;
            
            norm_tangt.push_back(EdgeNormalDirectionPoint(
                                       ortho_pt1, ortho_pt2,
                                       edge_pt - edge_tngt,
                                       edge_pt + edge_tngt));


            // cv::line(img, ortho_pt1, ortho_pt2, cv::Scalar(0, 255,
            // 0), 1);
            cv::line(img, cv::Point2f(x1, y1) + edge_pt,
                     edge_pt + cv::Point2f(x2, y2), cv::Scalar(0, 255, 0), 1);
            cv::line(img, edge_pt + edge_tngt, edge_pt -  edge_tngt,
                     cv::Scalar(255, 0, 255), 1);
          }
      }
       tangents.push_back(tangent);
       normal_points.push_back(norm_tangt);
    }
    cv::imshow("tangent", img);
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

void InteractiveSegmentation::mlsSmoothPointCloud(
    const pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<PointT>::Ptr scloud,
    pcl::PointCloud<pcl::Normal>::Ptr normals) {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mls_points(
        new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    pcl::MovingLeastSquares<PointT, pcl::PointXYZRGBNormal> mls;
    mls.setComputeNormals(true);
    mls.setInputCloud(cloud);
    mls.setPolynomialFit(true);
    mls.setSearchMethod(tree);
    mls.setSearchRadius(0.03);
    mls.process(*mls_points);
    for (int i = 0; i < mls_points->size(); i++) {
      pcl::Normal norm;
      norm.normal_x = mls_points->points[i].normal_x;
      norm.normal_y = mls_points->points[i].normal_y;
      norm.normal_z = mls_points->points[i].normal_z;
      PointT pt;
      pt.x = mls_points->points[i].x;
      pt.y = mls_points->points[i].y;
      pt.z = mls_points->points[i].z;
      pt.r = mls_points->points[i].r;
      pt.g = mls_points->points[i].g;
      pt.b = mls_points->points[i].b;
      normals->push_back(norm);
      scloud->push_back(pt);
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "interactive_segmentation");
    InteractiveSegmentation is;
    ros::spin();
    return 0;
}
