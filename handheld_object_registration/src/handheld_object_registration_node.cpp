
#include <handheld_object_registration/handheld_object_registration.h>

HandheldObjectRegistration::HandheldObjectRegistration():
    num_threads_(16), is_init_(false), min_points_size_(100),
    weight_decay_factor_(0.6f), init_weight_(1.0f) {
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    this->target_points_ = pcl::PointCloud<PointNormalT>::Ptr(
       new pcl::PointCloud<PointNormalT>);
    this->model_weights_.clear();
    
    //! temporary
    this->rendering_cuboid_ = boost::shared_ptr<jsk_msgs::BoundingBox>(
       new jsk_msgs::BoundingBox);
    this->rendering_cuboid_->pose.position.x = 0.0;
    this->rendering_cuboid_->pose.position.y = 0.0;
    this->rendering_cuboid_->pose.position.z = 1.0;
    this->rendering_cuboid_->pose.orientation.x = 0.0;
    this->rendering_cuboid_->pose.orientation.y = 0.0;
    this->rendering_cuboid_->pose.orientation.z = 0.0;
    this->rendering_cuboid_->pose.orientation.w = 1.0;
    this->rendering_cuboid_->dimensions.x = 0.5;
    this->rendering_cuboid_->dimensions.y = 0.5;
    this->rendering_cuboid_->dimensions.z = 0.5;
    
    this->onInit();
}

void HandheldObjectRegistration::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/handheld_object_registration/output/cloud", 1);
    this->pub_icp_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/handheld_object_registration/output/icp", 1);
    this->pub_bbox_ = this->pnh_.advertise<jsk_msgs::BoundingBoxArray>(
       "/handheld_object_registration/output/render_box", 1);
}

void HandheldObjectRegistration::subscribe() {
    this->screen_pt_ = this->pnh_.subscribe(
       "input_point", 1, &HandheldObjectRegistration::screenCB, this);
   
   
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_cinfo_.subscribe(this->pnh_, "input_cinfo", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(this->sub_cloud_, this->sub_cinfo_);
    this->sync_->registerCallback(
       boost::bind(&HandheldObjectRegistration::cloudCB, this, _1, _2));
}

void HandheldObjectRegistration::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_cinfo_.unsubscribe();
}

/**
 * temp for dev
 */
void HandheldObjectRegistration::screenCB(
    const geometry_msgs::PointStamped::ConstPtr &screen_msg) {
    this->screen_msg_ = *screen_msg;
    is_init_ = true;
}

void HandheldObjectRegistration::cloudCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::CameraInfo::ConstPtr &cinfo_msg) {
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty() || !is_init_) {
       ROS_ERROR_ONCE("-Input cloud is empty in callback");
       return;
    }
    ROS_INFO("\033[34m - Running Callback ... \033[0m");
    
    this->camera_info_ = cinfo_msg;
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    this->kdtree_->setInputCloud(cloud);

    std::cout << "kdtree made"  << "\n";
    
    PointNormal::Ptr normals(new PointNormal);
    this->getNormals(normals, cloud);
    
    PointCloud::Ptr region_cloud(new PointCloud);
    PointNormal::Ptr region_normal(new PointNormal);
    this->seedRegionGrowing(region_cloud, region_normal, screen_msg_,
                            cloud, normals);

    std::cout << "region growing done"  << "\n";
    
    //! delete this later
    pcl::PointCloud<PointNormalT>::Ptr src_points(
       new pcl::PointCloud<PointNormalT>);
    for (int i = 0; i < region_cloud->size(); i++) {
       PointNormalT pt;
       pt.x = region_cloud->points[i].x;
       pt.y = region_cloud->points[i].y;
       pt.z = region_cloud->points[i].z;
       pt.r = region_cloud->points[i].r;
       pt.g = region_cloud->points[i].g;
       pt.b = region_cloud->points[i].b;
       pt.normal_x = region_normal->points[i].normal_x;
       pt.normal_y = region_normal->points[i].normal_y;
       pt.normal_z = region_normal->points[i].normal_z;
       src_points->push_back(pt);
    }
    
    if (!this->target_points_->empty()) {

       std::cout << "updating"  << "\n";
       
       Eigen::Matrix<float, 4, 4> transformation;
       pcl::PointCloud<PointNormalT>::Ptr align_points(
          new pcl::PointCloud<PointNormalT>);

       std::cout << "icp"  << "\n";
       
       if (!this->registrationICP(align_points, transformation, src_points)) {
          ROS_ERROR("- ICP cannot converge.. skipping");
          return;
       }

       pcl::PointCloud<PointNormalT>::Ptr tmp_cloud(
          new pcl::PointCloud<PointNormalT>);
       // PointCloud::Ptr tmp_cloud (new PointCloud);
       pcl::transformPointCloud(*src_points, *tmp_cloud, transformation);

       // std::cout << "\n " << transformation  << "\n";

       this->modelUpdate(src_points, target_points_, transformation);
       
       sensor_msgs::PointCloud2 ros_cloud;
       // pcl::toROSMsg(*tmp_cloud, ros_cloud);
       pcl::toROSMsg(*align_points, ros_cloud);
       ros_cloud.header = cloud_msg->header;
       this->pub_icp_.publish(ros_cloud);
    } else {
       this->target_points_->clear();
       *target_points_ = *src_points;

       this->model_weights_.resize(static_cast<int>(target_points_->size()));
       for (int i = 0; i < this->target_points_->size(); i++) {
          this->model_weights_[i].weight = this->init_weight_;
       }
    }

    std::cout << region_cloud->size()  << "\n";
    ROS_INFO("Done Processing");
    
    sensor_msgs::PointCloud2 *ros_cloud = new sensor_msgs::PointCloud2;
    // pcl::toROSMsg(*region_cloud, *ros_cloud);
    pcl::toROSMsg(*target_points_, *ros_cloud);
    ros_cloud->header = cloud_msg->header;
    this->pub_cloud_.publish(*ros_cloud);
    
    jsk_msgs::BoundingBoxArray *rviz_bbox = new jsk_msgs::BoundingBoxArray;
    this->rendering_cuboid_->header = cloud_msg->header;
    rviz_bbox->boxes.push_back(*rendering_cuboid_);
    rviz_bbox->header = cloud_msg->header;
    this->pub_bbox_.publish(*rviz_bbox);

    delete ros_cloud;
    delete rviz_bbox;
    pcl::PointCloud<PointNormalT>().swap(*src_points);
    PointCloud().swap(*cloud);
    PointCloud().swap(*region_cloud);
    PointNormal().swap(*region_normal);
    
    // is_init_ = false;
}

void HandheldObjectRegistration::modelUpdate(
    pcl::PointCloud<PointNormalT>::Ptr src_points,
    pcl::PointCloud<PointNormalT>::Ptr target_points,
    const Eigen::Matrix<float, 4, 4> transformation) {
    if (src_points->empty() || target_points->empty()) {
       ROS_ERROR("Empty input points for update");
       return;
    }
    // TODO(MIN_SIZE_CHECK):
    
    pcl::PointCloud<PointNormalT>::Ptr trans_points(
       new pcl::PointCloud<PointNormalT>);
    Eigen::Matrix<float, 4, 4> transformation_inv = transformation.inverse();
    // pcl::transformPointCloud(*src_points, *trans_points,
    // transformation);
    
    pcl::transformPointCloud(*target_points, *trans_points, transformation_inv);
    
    cv::Mat src_image;
    cv::Mat src_depth;
    cv::Mat src_indices = this->project3DTo2DDepth(
       src_image, src_depth, src_points/*trans_points*/);
    
    //! move it out***
    cv::Mat target_image;
    cv::Mat target_depth;
    cv::Mat target_indices = this->project3DTo2DDepth(
       target_image, target_depth, trans_points/*target_points*/);
    
    const float DISTANCE_THRESH_ = 0.10f;
    const float ANGLE_THRESH_ = M_PI/3.0f;

    pcl::PointCloud<PointNormalT>::Ptr update_model(
       new pcl::PointCloud<PointNormalT>);
    // *update_model += *target_points;
    *update_model += *trans_points;
    
    // update_model->resize(target_points->size() + src_points->size());
    
    int icounter = 0;
    for (int j = 0; j < target_image.rows; j++) {
       for (int i = 0; i < target_image.cols; i++) {
          float t_depth = target_depth.at<float>(j, i);
          float s_depth = src_depth.at<float>(j, i);
          int s_index = src_indices.at<int>(j, i);
          int t_index = target_indices.at<int>(j, i);

          //! debug only
          if (t_index >= static_cast<int>(target_points->size()) ||
              s_index >= static_cast<int>(src_points->size())) {
             ROS_ERROR("- index out of size");
             std::cout << s_index << "\t" << src_points->size()  << "\n";
             std::cout << t_index << "\t" << target_points->size()  << "\n";
          }
          
          if (s_depth != 0.0f && t_depth != 0.0f) {  //! common points
             // TODO(FILL PREV NAN):
             Eigen::Vector4f s_norm = src_points->points[
                s_index].getNormalVector4fMap();
             Eigen::Vector4f t_norm = target_points->points[
                t_index].getNormalVector4fMap();
             float angle = std::acos(s_norm.dot(t_norm) /
                                     (s_norm.norm() * t_norm.norm()));
             float dist = std::abs(t_depth - s_depth);
             if (dist < DISTANCE_THRESH_ && angle < ANGLE_THRESH_) {
                //! common point keep previous points
                // update_model->push_back(target_points->points[t_index]);
                // update_model->push_back(trans_points->points[t_index]);
             } else {
                model_weights_[t_index].weight *= this->weight_decay_factor_;
             }
          } else if (s_depth != 0 && t_depth == 0) {  //! new points
             // update_model->push_back(trans_points->points[s_index]);
             update_model->push_back(src_points->points[s_index]);

             Model m_weight;
             m_weight.weight = this->init_weight_;
             this->model_weights_.push_back(m_weight);
          } else if (s_depth == 0 && t_depth != 0) {  //! new points
             // update_model->push_back(target_points->points[t_index]);
             // update_model->push_back(trans_points->points[t_index]);
             model_weights_[t_index].weight *= this->weight_decay_factor_;
          }
       }
    }

    std::cout << "\033[35m - INFO: \033[0m" << update_model->size()
              << "\t" << this->model_weights_.size() << "\n";

    this->target_points_->clear();
    std::vector<Model> temp_model_weight;
    for (int i = 0; i < model_weights_.size(); i++) {
       if (this->model_weights_[i].weight > 0.5) {
          target_points_->push_back(update_model->points[i]);
          temp_model_weight.push_back(this->model_weights_[i]);
       }
    }
    this->model_weights_.clear();
    this->model_weights_ = temp_model_weight;
    
    // pcl::transformPointCloud(*update_model, *target_points_, transformation);
    pcl::transformPointCloud(*target_points_, *target_points_, transformation);
    // pcl::copyPointCloud<PointNormalT, PointNormalT>(
    //    *update_model, *target_points_);

    
    cv::namedWindow("trans_points", cv::WINDOW_NORMAL);
    cv::namedWindow("target", cv::WINDOW_NORMAL);
    cv::imshow("trans_points", src_image);
    cv::imshow("target", target_image);

    // cv::cvtColor(src_image, src_image, CV_BGR2GRAY);
    // cv::cvtColor(target_image, target_image, CV_BGR2GRAY);
    // cv::Mat diff = src_image - target_image;
    // cv::imshow("diff", diff);
    
    cv::waitKey(3);
}

bool HandheldObjectRegistration::registrationICP(
    const pcl::PointCloud<PointNormalT>::Ptr align_points,
    Eigen::Matrix<float, 4, 4> &transformation,
    const pcl::PointCloud<PointNormalT>::Ptr src_points) {
    if (src_points->empty()) {
       ROS_ERROR("- ICP failed. Incorrect inputs sizes ");
       return false;
    }
    pcl::IterativeClosestPointWithNormals<PointNormalT, PointNormalT>::Ptr icp(
       new pcl::IterativeClosestPointWithNormals<PointNormalT, PointNormalT>);
    icp->setMaximumIterations(10);
    icp->setInputSource(src_points);
    icp->setInputTarget(this->target_points_);
    icp->align(*align_points);
    transformation = icp->getFinalTransformation();

    std::cout << "has converged:" << icp->hasConverged() << " score: "
              << icp->getFitnessScore() << std::endl;

    return (icp->hasConverged() == 1) ? true : false;
}


bool HandheldObjectRegistration::seedRegionGrowing(
    PointCloud::Ptr out_cloud, PointNormal::Ptr out_normals,
    const geometry_msgs::PointStamped point,
    const PointCloud::Ptr cloud, PointNormal::Ptr normals) {
    if (cloud->empty() || normals->size() != cloud->size()) {
       ROS_ERROR("- Region growing failed. Incorrect inputs sizes ");
       return false;
    }
    int seed_index  = point.point.x + (640 * point.point.y);
    PointT seed_point = cloud->points[seed_index];
    if (isnan(seed_point.x) || isnan(seed_point.y) || isnan(seed_point.z)) {
       ROS_ERROR("- Seed Point is Nan. Skipping");
       return false;
    }
    
    const int in_dim = static_cast<int>(cloud->size());
    int *labels = reinterpret_cast<int*>(malloc(sizeof(int) * in_dim));
    for (int i = 0; i < in_dim; i++) {
       if (i == seed_index) {
          labels[i] = 1;
       }
       labels[i] = -1;
    }

    this->seedCorrespondingRegion(labels, cloud, normals,
                                  seed_index, seed_index);
    out_cloud->clear();
    out_normals->clear();
    for (int i = 0; i < in_dim; i++) {
       if (labels[i] != -1) {
          out_cloud->push_back(cloud->points[i]);
          out_normals->push_back(normals->points[i]);
       }
    }
    
    free(labels);
    return true;
}

void HandheldObjectRegistration::seedCorrespondingRegion(
    int *labels, const PointCloud::Ptr cloud, const PointNormal::Ptr normals,
    const int parent_index, const int seed_index) {
    Eigen::Vector4f seed_point = cloud->points[seed_index].getVector4fMap();
    Eigen::Vector4f seed_normal = normals->points[
       seed_index].getNormalVector4fMap();
   
    std::vector<int> neigbor_indices;
    this->getPointNeigbour<int>(neigbor_indices, cloud,
                                cloud->points[parent_index],
                                18);

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
            // Eigen::Vector4f parent_norm = normals->points[
            //     parent_index].getNormalVector4fMap();
            Eigen::Vector4f child_pt = cloud->points[index].getVector4fMap();
            Eigen::Vector4f child_norm = normals->points[
                index].getNormalVector4fMap();
            if (this->seedVoxelConvexityCriteria(
                   seed_point, seed_normal, parent_pt, child_pt,
                   child_norm, -0.01f) == 1) {
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
           seedCorrespondingRegion(labels, cloud, normals, index, seed_index);
        }
    }
}


int HandheldObjectRegistration::seedVoxelConvexityCriteria(
    Eigen::Vector4f seed_point, Eigen::Vector4f seed_normal,
    Eigen::Vector4f c_centroid, Eigen::Vector4f n_centroid,
    Eigen::Vector4f n_normal, const float thresh) {
    float im_relation = (n_centroid - c_centroid).dot(n_normal);
    float pt2seed_relation = FLT_MAX;
    float seed2pt_relation = FLT_MAX;
    pt2seed_relation = (n_centroid - seed_point).dot(n_normal);
    seed2pt_relation = (seed_point - n_centroid).dot(seed_normal);
    if (seed2pt_relation > thresh && pt2seed_relation > thresh) {
      return 1;
    } else {
       return -1;
    }
}

void HandheldObjectRegistration::getNormals(
    PointNormal::Ptr normals, const PointCloud::Ptr cloud) {
    if (cloud->empty()) {
       ROS_ERROR("-Input cloud is empty in normal estimation");
       return;
    }
    
    pcl::IntegralImageNormalEstimation<PointT, NormalT> ne;
    ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setInputCloud(cloud);
    ne.compute(*normals);
}

template<class T>
void HandheldObjectRegistration::getPointNeigbour(
    std::vector<int> &neigbor_indices, const PointCloud::Ptr cloud,
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

cv::Mat HandheldObjectRegistration::project3DTo2DDepth(
    cv::Mat &image, cv::Mat &depth_image,
    const pcl::PointCloud<PointNormalT>::Ptr cloud, const float max_distance) {
    if (cloud->empty()) {
       ROS_ERROR("- Empty cloud. Cannot project 3D to 2D depth.");
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
       K[i] = camera_info_->K[i];
       R[i] = camera_info_->R[i];
    }

    cv::Mat camera_matrix = cv::Mat(3, 3, CV_32F, K);
    cv::Mat rotationMatrix = cv::Mat(3, 3, CV_32F, R);
    float tvec[3];
    tvec[0] = camera_info_->P[3];
    tvec[1] = camera_info_->P[7];
    tvec[2] = camera_info_->P[11];
    cv::Mat translation_matrix = cv::Mat(3, 1, CV_32F, tvec);
    
    float D[5];
    for (int i = 0; i < 5; i++) {
       D[i] = camera_info_->D[i];
    }
    cv::Mat distortion_model = cv::Mat(5, 1, CV_32F, D);
    cv::Mat rvec;
    cv::Rodrigues(rotationMatrix, rvec);
    
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(object_points, rvec, translation_matrix,
                      camera_matrix, distortion_model, image_points);
    image = cv::Mat::zeros(camera_info_->height, camera_info_->width, CV_8UC3);
    depth_image = cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat indices = cv::Mat(image.size(), CV_32S);
    // indices = cv::Scalar(-1);
    
    for (int j = 0; j < image.rows; j++) {
       for (int i = 0; i < image.cols; i++) {
          indices.at<int>(j, i) = -1;
       }
    }
    
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

          depth_image.at<float>(y, x) = cloud->points[i].z / max_distance;
          indices.at<int>(y, x) = i;
       }
    }

    //! debug
    /*
    for (int i = 0; i < indices.rows; i++) {
       for (int j = 0; j < indices.cols; j++) {
          int ind = indices.at<int>(i, j);
          if (ind >= static_cast<int>(cloud->size())) {
             std::cout << "\033[34m Bigger: " << ind  << "\t" << j
                       << ", " << i << "\t" << cloud->size() << "\033[0m\n";
          }
       }
    }
    */
    return indices;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "handheld_object_registration");
    HandheldObjectRegistration hor;
    ros::spin();
    return 0;
}

