
#include <handheld_object_registration/handheld_object_registration.h>

HandheldObjectRegistration::HandheldObjectRegistration():
    num_threads_(16), is_init_(false), min_points_size_(100),
    weight_decay_factor_(0.75f), init_weight_(1.0f), pose_flag_(false) {
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    
    this->target_points_ = pcl::PointCloud<PointNormalT>::Ptr(
       new pcl::PointCloud<PointNormalT>);
    this->initial_points_ = pcl::PointCloud<PointNormalT>::Ptr(
       new pcl::PointCloud<PointNormalT>);
    this->prev_points_ = pcl::PointCloud<PointNormalT>::Ptr(
       new pcl::PointCloud<PointNormalT>);
    this->input_cloud_ = PointCloud::Ptr(new PointCloud);
    this->input_normals_ = PointNormal::Ptr(new PointNormal);
    
    this->orb_gpu_ = cv::cuda::ORB::create(1000, 1.20f, 8, 11, 0, 2,
                                           cv::ORB::HARRIS_SCORE, 31, true);
    // this->voxel_weights_.clear();
    this->prev_transform_ = Eigen::Affine3f::Identity();
    this->update_counter_ = 0;
    this->initial_transform_ = Eigen::Matrix4f::Identity();
    this->transformation_cache_.clear();
    
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
    this->pub_templ_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/handheld_object_registration/output/template", 1);
    this->pub_bbox_ = this->pnh_.advertise<jsk_msgs::BoundingBoxArray>(
       "/handheld_object_registration/output/render_box", 1);
}

void HandheldObjectRegistration::subscribe() {
    this->screen_pt_ = this->pnh_.subscribe(
       "input_point", 1, &HandheldObjectRegistration::screenCB, this);
    this->pf_pose_ = this->pnh_.subscribe(
       "input_pose", 1, &HandheldObjectRegistration::poseCB, this);
   
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

/**
 * temp here:: later move to cloudCB
 */
void HandheldObjectRegistration::poseCB(
    const geometry_msgs::PoseStamped::ConstPtr &pose_msg) {
    this->pose_msg_ = pose_msg;
    this->pose_flag_ = true;
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
    
    //! timer
    struct timeval timer_start, timer_end;
    // gettimeofday(&timer_start, NULL);

    this->camera_info_ = cinfo_msg;
    
    PointNormal::Ptr normals(new PointNormal);
    this->getNormals(normals, cloud);

    int seed_index  = screen_msg_.point.x +
       (camera_info_->width * screen_msg_.point.y);
    PointT seed_point = cloud->points[seed_index];
    if (this->pose_flag_) {
       seed_point.x = pose_msg_->pose.position.x;
       seed_point.y = pose_msg_->pose.position.y;
       seed_point.z = pose_msg_->pose.position.z;

       double d = pcl::distances::l2(seed_point.getVector4fMap(),
                          prev_seed_point_.getVector4fMap());
       
       this->prev_seed_point_ = seed_point;

       std::cout  << "\n";
       std::cout << seed_point  << "\t Dist: " << d << "\n";
       std::cout << pose_msg_->pose.position.x << ", " <<
          pose_msg_->pose.position.y << ", " <<
          pose_msg_->pose.position.z << "\n----------------\n";

    }
    if (isnan(seed_point.x) || isnan(seed_point.y) ||isnan(seed_point.z)) {
       ROS_ERROR("SEED POINT IS NAN");
       return;
    }

    pcl::PointCloud<PointNormalT>::Ptr src_points(
       new pcl::PointCloud<PointNormalT>);
    getObjectRegion(src_points, cloud, normals, seed_point);
    

    // this->seedRegionGrowing(src_points, seed_point, cloud, normals);

    /**
     * DEBUG
    std::clock_t start;
    start = std::clock();

    float equation[4];
    this->symmetricPlane(equation, src_points);
    
    double duration = (std::clock() - start) /
       static_cast<double>(CLOCKS_PER_SEC);
    std::cout << "printf: " << duration <<'\n';

    sensor_msgs::PointCloud2 ros_cloud1;
    pcl::toROSMsg(*src_points, ros_cloud1);
    ros_cloud1.header = cloud_msg->header;
    this->pub_icp_.publish(ros_cloud1);
    
    return;
    * DEBUG
    */


    /**
     * update to current tracker pose
     */
    Eigen::Affine3f tracker_transform = Eigen::Affine3f::Identity();
    if (this->pose_flag_) {
       geometry_msgs::PoseStamped::ConstPtr pose_msg = pose_msg_;
       Eigen::Affine3f transform_model = Eigen::Affine3f::Identity();
       transform_model.translation() <<
          pose_msg->pose.position.x,
          pose_msg->pose.position.y,
          pose_msg->pose.position.z;
       Eigen::Quaternion<float> pf_quat = Eigen::Quaternion<float>(
          pose_msg->pose.orientation.w, pose_msg->pose.orientation.x,
          pose_msg->pose.orientation.y, pose_msg->pose.orientation.z);
       transform_model.rotate(pf_quat);
       
       if (!this->prev_points_->empty()) {
          tracker_transform =  transform_model * prev_transform_.inverse();
          // pcl::transformPointCloudWithNormals(*prev_points_,
          //                          *prev_points_, tracker_transform);
          pcl::transformPointCloudWithNormals(*target_points_,
                                              *target_points_,
                                              tracker_transform);
          this->pose_flag_ = false;
       }
       this->prev_transform_ = transform_model;
    } else {
       ROS_WARN("TRACKER NOT SET");
    }
    /**
     * END
     */
    
    
    if (!this->target_points_->empty()) {
       std::cout << "updating"  << "\n";

       //! timer
       struct timeval timer_start, timer_end;
       gettimeofday(&timer_start, NULL);
       
       //! this->modelUpdate(src_points, target_points_);

       //! timer
       gettimeofday(&timer_end, NULL);
       double delta = ((timer_end.tv_sec  - timer_start.tv_sec) * 1000000u +
                       timer_end.tv_usec - timer_start.tv_usec) / 1.e6;
       ROS_ERROR("TIME: %3.6f, %d", delta, target_points_->size());
       
       sensor_msgs::PointCloud2 ros_cloud;
       pcl::toROSMsg(*src_points, ros_cloud);
       ros_cloud.header = cloud_msg->header;
       this->pub_icp_.publish(ros_cloud);
    } else {
       this->target_points_->clear();
       pcl::copyPointCloud<PointNormalT, PointNormalT>(*src_points,
                                                       *target_points_);
       pcl::copyPointCloud<PointNormalT, PointNormalT>(*src_points,
                                                       *initial_points_);
       //! initial user marked region
       this->project3DTo2DDepth(initial_projection_, target_points_);

       //! deform the initial region slightly
       cv::Mat mask = cv::Mat::zeros(480, 640, CV_8UC1);
       for (int i = initial_projection_.y; i < initial_projection_.y +
               initial_projection_.height; i++) {
          for (int j = initial_projection_.x; j < initial_projection_.x +
                  initial_projection_.width; j++) {
             if (initial_projection_.indices.at<int>(i, j) != -1) {
                mask.at<uchar>(i, j) = 255;
             }
          }
       }
       cv::imshow("mask", mask);
       
       cv::waitKey(3);


       //! init weight
       /*
       this->point_weights_.clear();
       this->point_weights_.resize(static_cast<int>(target_points_->size()));

       this->voxel_weights_.clear();
       this->voxel_weights_.resize(static_cast<int>(target_points_->size()));
       for (int i = 0; i < target_points_->size(); i++) {
          point_weights_[i] = this->init_weight_;
          int j = 0;
          this->voxel_weights_[i].weight[j] = this->init_weight_;
          for (j = 1; j < HISTORY_WINDOW; j++) {
             this->voxel_weights_[i].weight[j] = -1;
          }
       }
       */
    }

    this->prev_points_->clear();
    pcl::copyPointCloud<PointNormalT, PointNormalT>(*src_points,
                                                    *prev_points_);
    this->project3DTo2DDepth(this->prev_projection_, this->prev_points_);
    
    
    std::cout << src_points->size() << "\t"
              << target_points_->size()  << "\n";
    ROS_INFO("Done Processing");
    

    //! publish data
    sensor_msgs::PointCloud2 *ros_cloud = new sensor_msgs::PointCloud2;
    pcl::toROSMsg(*target_points_, *ros_cloud);
    ros_cloud->header = cloud_msg->header;
    this->pub_cloud_.publish(*ros_cloud);

    /*
    if (update_counter_++ == 1) {
       sensor_msgs::PointCloud2 *ros_templ = new sensor_msgs::PointCloud2;
       pcl::toROSMsg(*region_cloud, *ros_templ);
       ros_templ->header = cloud_msg->header;
       this->pub_templ_.publish(*ros_templ);
       this->update_counter_ = 0;
    }
    */
    
    
    jsk_msgs::BoundingBoxArray *rviz_bbox = new jsk_msgs::BoundingBoxArray;
    this->rendering_cuboid_->header = cloud_msg->header;
    rviz_bbox->boxes.push_back(*rendering_cuboid_);
    rviz_bbox->header = cloud_msg->header;
    this->pub_bbox_.publish(*rviz_bbox);
    
    delete ros_cloud;
    delete rviz_bbox;
    pcl::PointCloud<PointNormalT>().swap(*src_points);
    PointCloud().swap(*cloud);
    // PointCloud().swap(*region_cloud);
    PointNormal().swap(*normals);
    
    // is_init_ = false;
}


void HandheldObjectRegistration::modelUpdate(
    pcl::PointCloud<PointNormalT>::Ptr src_points,
    pcl::PointCloud<PointNormalT>::Ptr target_points) {
    if (src_points->empty() || target_points->empty()) {
       ROS_ERROR("Empty input points for update");
       return;
    }

    pcl::PointCloud<PointNormalT>::Ptr update_points(
       new pcl::PointCloud<PointNormalT>);
    
    // TODO(MIN_SIZE_CHECK):
    ROS_INFO("\033[33m PROJECTION TO 2D \033[0m");
    
    ProjectionMap src_projection;
    this->project3DTo2DDepth(src_projection, src_points);

    //! move it out***
    ProjectionMap target_projection = this->prev_projection_;
    // this->project3DTo2DDepth(target_projection, this->prev_points_);

    
    //! fitness check
    float benchmark_fitness = this->checkRegistrationFitness(
       src_projection, src_points, target_projection, prev_points_);
    ROS_INFO("\033[34m BENCHMARK: %3.2f \033[0m", benchmark_fitness);

    
    //!  Feature check
    std::vector<CandidateIndices> candidate_indices;
    this->featureBasedTransformation(candidate_indices,
                                     this->prev_points_,
                                     target_projection.indices,
                                     target_projection.rgb,
                                     src_points, src_projection.indices,
                                     src_projection.rgb);
    
    ROS_INFO("\033[33m COMPUTING TRANSFORMATION \033[0m");
    
    //! compute transformation
    Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity();
    if (candidate_indices.size() > 7) {
       float energy = FLT_MAX;
       Eigen::Matrix3f rotation;
       Eigen::Vector3f transl;

       PointCloud::Ptr src_cloud(new PointCloud);
       PointCloud::Ptr target_cloud(new PointCloud);

       pcl::Correspondences correspondences;
       for (int i = 0; i < candidate_indices.size(); i++) {
          int src_index = candidate_indices[i].source_index;
          int tgt_index = candidate_indices[i].target_index;
          if (src_index != -1 && tgt_index != -1) {
             PointNormalT spt = src_points->points[src_index];
             PointNormalT tpt = this->prev_points_->points[tgt_index];

             pcl::Correspondence corr;
             corr.index_query = tgt_index;
             corr.index_match = src_index;
             correspondences.push_back(corr);
             
             PointT pt;
             pt.x = spt.x; pt.y = spt.y; pt.z = spt.z;
             src_cloud->push_back(pt);
             pt.x = tpt.x; pt.y = tpt.y; pt.z = tpt.z;
             target_cloud->push_back(pt);
          }
       }
       if (!src_cloud->empty() || !target_cloud->empty()) {
          /*
          pcl::registration::TransformationEstimationSVD<
             PointT, PointT> transformation_estimation;
          transformation_estimation.estimateRigidTransformation(
             *target_cloud, *src_cloud, correspondences, transform_matrix);
             */
          int iter = 0;
          while (iter++ < 1) {
             pcl::registration::TransformationEstimationSVD<
                PointNormalT, PointNormalT> transformation_estimation;
             Eigen::Matrix4f trans_mat = Eigen::Matrix4f::Identity();
             transformation_estimation.estimateRigidTransformation(
                *prev_points_, *src_points, correspondences, trans_mat);
             transformPointCloudWithNormalsGPU(prev_points_, update_points,
                                               trans_mat);
             this->prev_points_->clear();
             pcl::copyPointCloud<PointNormalT, PointNormalT>(
                *update_points, *prev_points_);
             
             transform_matrix = transform_matrix * trans_mat;
          }
       }
       transformPointCloudWithNormalsGPU(target_points, target_points,
                                         transform_matrix);
    }

    //! get current visible points on the model
    update_points->clear();
    this->project3DTo2DDepth(target_projection, target_points);
    for (int i = 0; i < target_points->size(); i++) {
       if (target_projection.visibility_flag[i]) {
          update_points->push_back(target_points->points[i]);
       }
    }
    if (update_points->empty()) {
       pcl::copyPointCloud<PointNormalT, PointNormalT>(*target_points,
                                                       *update_points);
    }
    
    ROS_INFO("\033[33m PCL--ICP \033[0m");
    pcl::PointCloud<PointNormalT>::Ptr aligned_points(
       new pcl::PointCloud<PointNormalT>);
    Eigen::Matrix<float, 4, 4> icp_transform = Eigen::Matrix4f::Identity();
    this->registrationICP(aligned_points, icp_transform,
                          update_points, src_points);
    Eigen::Matrix4f final_transformation = icp_transform * transform_matrix;
    
    
    //! transform the model to current orientation
    transformPointCloudWithNormalsGPU(target_points, target_points,
                                      icp_transform);

    this->project3DTo2DDepth(target_projection, target_points);
    this->modelVoxelUpdate(target_points, target_projection,
                           src_points, src_projection);
    
    // this->project3DTo2DDepth(target_projection, aligned_points);
    // this->modelVoxelUpdate(aligned_points, target_projection,
    //                        src_points, src_projection);

    
    //! transform to init
    this->initial_transform_ =  final_transformation * initial_transform_;
    pcl::PointCloud<PointNormalT>::Ptr init_trans_points(
       new pcl::PointCloud<PointNormalT>);
    Eigen::Matrix4f inv = this->initial_transform_.inverse();
    transformPointCloudWithNormalsGPU(target_points, init_trans_points, inv);

    // TODO(THRESHOLD):  if above certain value than run icp

    //! realignment for error compensation
    /*
    Eigen::Matrix4f transform_error;
    this->registrationICP(init_trans_points, transform_error,
                          init_trans_points, this->initial_points_);
    transformPointCloudWithNormalsGPU(target_points, target_points,
                                      transform_error);
    */
    
    
    ProjectionMap target_project_initial;
    this->project3DTo2DDepth(target_project_initial, init_trans_points);
    
    //! plot outliers
    cv::Mat outlier = cv::Mat::zeros(this->camera_info_->height,
                                     this->camera_info_->width,  CV_8UC3);

    for (int j = 0; j < this->initial_projection_.indices.rows; j++) {
       for (int i = 0;  i < this->initial_projection_.indices.cols; i++) {
          int tpi_index = target_project_initial.indices.at<int>(j, i);
          int ip_index = initial_projection_.indices.at<int>(j, i);
          
          if (tpi_index!= -1 && ip_index == -1) {
             // TODO(CHECK):  check distance and angle
             target_points->points[tpi_index].x =
                std::numeric_limits<float>::quiet_NaN();
             target_points->points[tpi_index].y =
                std::numeric_limits<float>::quiet_NaN();
             target_points->points[tpi_index].z =
                std::numeric_limits<float>::quiet_NaN();
                
             outlier.at<cv::Vec3b>(j, i)[2] = 255;
          } else if (ip_index != -1 && tpi_index == -1) {
             outlier.at<cv::Vec3b>(j, i)[1] = 255;
          } else if (ip_index != -1 && tpi_index != -1) {
             outlier.at<cv::Vec3b>(j, i) = initial_projection_.rgb.at<
                cv::Vec3b>(j, i);
          }
       }
    }

    update_points->clear();
    pcl::copyPointCloud<PointNormalT, PointNormalT>(
       *target_points, *update_points);
    target_points->clear();
    for (int i = 0; i < update_points->size(); i++) {
       PointNormalT pt = update_points->points[i];
       if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z) &&
           !isnan(pt.normal_x) && !isnan(pt.normal_y) && !isnan(pt.normal_z)) {

          target_points->push_back(update_points->points[i]);
       }
    }

    ROS_ERROR("FINAL UPDATE: %d", target_points->size());
    
    cv::imshow("intial", initial_projection_.rgb);
    cv::imshow("transfromed", target_project_initial.rgb);
    cv::imshow("outlier", outlier);
    cv::waitKey(3);
    return;

    
    if (this->update_counter_++ > 2) {
       
       for (int j = src_projection.y;
            j < src_projection.height + src_projection.y; j++) {
          for (int i = src_projection.x;
               i < src_projection.width + src_projection.x; i++) {
             /*
             //! matching to the init model
             int it_index = target_project_initial.indices.at<int>(j, i);
             int ip_index = this->initial_projection_.indices.at<int>(j, i);

             if (it_index != -1) {
                //! check the normal deviation
                Eigen::Vector4f it_norm = init_trans_points->points[
                   it_index].getNormalVector4fMap().normalized();
                Eigen::Vector4f ip_norm = this->initial_points_->points[
                   ip_index].getNormalVector4fMap().normalized();

                float angle = std::acos(ip_norm.dot(it_norm));
                                
                if (angle < M_PI/18 && !isnan(angle)) {
                   init_outlier_flag[it_index] = false;
                   outlier.at<cv::Vec3b>(j, i)[2] = 255;
                } else {
                   init_outlier_flag[it_index] = true;
                   outlier.at<cv::Vec3b>(j, i)[1] = 255;
                }
       
             }
             if (it_index != -1 && ip_index == -1) {
                init_outlier_flag[it_index] = true;
                outlier.at<cv::Vec3b>(j, i)[1] = 255;
             }
             */
                   
                   
          }
       }
    }

    this->transformation_cache_.push_back(final_transformation);

    std::cout << "PRINTING"  << "\n";
    
    cv::waitKey(3);
    return;

    /**
     * END CPU ICP
     */


    

    
    // --> change name
    //! project the target points

    /*
    const int SEARCH_WSIZE = 8;
    const float ICP_DIST_THRESH = 0.05f;
    cv::Mat debug_im = cv::Mat::zeros(src_projection.rgb.size(), CV_8UC3);
    float cpu_energy = 0.0;
    pcl::Correspondences cpu_correspondences;
    for (int j = target_projection.y; j < target_projection.y +
            target_projection.height; j++) {
       for (int i = target_projection.x; i < target_projection.x +
               target_projection.width; i++) {
          
          int model_index = target_projection.indices.at<int>(j, i);
          if (model_index != -1) {
             pcl::Correspondence corr;
             int x = i - (SEARCH_WSIZE/2);
             int y = j - (SEARCH_WSIZE/2);
             if (this->conditionROI(x, y, SEARCH_WSIZE, SEARCH_WSIZE,
                                    target_projection.rgb.size())) {
                Eigen::Vector4f model_pt = target_points->points[
                   model_index].getVector4fMap();
                double min_dsm = std::numeric_limits<double>::max();
                int min_ism = -1;
                for (int l = y; l < y + SEARCH_WSIZE; l++) {
                   for (int k = x; k < x + SEARCH_WSIZE; k++) {
                      int src_index = src_projection.indices.at<int>(l, k);
                      Eigen::Vector4f src_pt = src_points->points[
                         src_index].getVector4fMap();
                      double dsm = pcl::distances::l2(src_pt, model_pt);
                      if (dsm < min_dsm && !isnan(dsm)) {
                         min_dsm = dsm;
                         min_ism = src_index;
                      }
                   }
                }
                if (min_ism != -1) {
                   if (min_dsm < ICP_DIST_THRESH) {
                      corr.index_match = min_ism;
                      corr.index_query = model_index;
                      cpu_correspondences.push_back(corr);

                      cpu_energy += min_dsm;
                   }
                }
             }
          }
       }
    }

    std::cout << "cpu energy: " << cpu_energy /
       static_cast<float>(cpu_correspondences.size()) << "\n";

    Eigen::Matrix4f cpu_trans = Eigen::Matrix4f::Identity();
    pcl::registration::TransformationEstimationPointToPlaneLLS<
       PointNormalT, PointNormalT> transformation_estimation;
    transformation_estimation.estimateRigidTransformation(
       *target_points, *src_points, cpu_correspondences, cpu_trans);

    std::cout << "------\n" << "CPU TRANS:"  << "\n";
    std::cout << "CPU SIZE: " << cpu_correspondences.size()  << "\n";
    std::cout << cpu_trans  << "\n-------------\n";
    //!------------------------
    */

    /* GPU BASED ALIGNMENT
    const float ENERGY_THRESH = 0.0005;
    const int MAX_ITER = 0;    
    Eigen::Matrix4f icp_trans = Eigen::Matrix4f::Identity();
    pcl::Correspondences correspondences;
    float energy;
    int icounter = 0;
    bool copy_src = true;
    while (true) {
       bool data_copied = allocateCopyDataToGPU(
          correspondences, energy, copy_src, src_points,
          src_projection, target_points, target_projection);
       if (data_copied) {
          std::cout << "ENERGY LEVEL:  " << energy  << "\n";

          if (correspondences.size() > 7) {
             pcl::registration::TransformationEstimationPointToPlaneLLS<
                PointNormalT, PointNormalT> transformation_estimation;
             transformation_estimation.estimateRigidTransformation(
                *target_points, *src_points, correspondences, icp_trans);
             transformPointCloudWithNormalsGPU(target_points, target_points,
                                               icp_trans);
          }
       }
       copy_src = true;

       if (icounter++ > MAX_ITER || energy < ENERGY_THRESH) {
          break;
       }
    }
    */

}

void HandheldObjectRegistration::modelVoxelUpdate(
    const pcl::PointCloud<PointNormalT>::Ptr target_points,
    const ProjectionMap target_projection,
    const pcl::PointCloud<PointNormalT>::Ptr src_points,
    const ProjectionMap src_projection) {
    if (src_points->empty() || target_points->empty()) {
       ROS_ERROR("INPUTS FOR MODELVOXELUPDATE ARE EMPTY");
       return;
    }
    
    pcl::PointCloud<PointNormalT>::Ptr update_model(
       new pcl::PointCloud<PointNormalT>);
    pcl::copyPointCloud<PointNormalT, PointNormalT>(
       *target_points, *update_model);
    
    const float DIST_THRESH = 0.01f;

    cv::Mat visz = cv::Mat::zeros(480, 640, CV_8UC3);
    
    for (int j = 0; j < src_projection.depth.rows; j++) {
       for (int i = 0; i < src_projection.depth.cols; i++) {
          float t_depth = target_projection.depth.at<float>(j, i);
          float s_depth = src_projection.depth.at<float>(j, i);
          
          int s_index = src_projection.indices.at<int>(j, i);
          int t_index = target_projection.indices.at<int>(j, i);
          
          // if ((t_index >= 0 && t_index <
          //      static_cast<int>(target_points->size())) &&
          //     (s_index >= 0 && s_index <
          //      static_cast<int>(src_points->size()))) {
             if (s_index != -1 && t_index != -1) {
                float dist = std::fabs(src_points->points[s_index].z -
                                       target_points->points[t_index].z);
                if (dist < DIST_THRESH) {
                   update_model->points[t_index] = src_points->points[s_index];

                   visz.at<cv::Vec3b>(j, i)[1] = 255;
                   
                } else {
                   // update_model->push_back(src_points->points[s_index]);
                }
             } else if (s_index == -1 && t_index != -1) {
                //! already in the model
                visz.at<cv::Vec3b>(j, i)[0] = 255;
             }  else if (s_index != -1 && t_index == -1) {
                update_model->push_back(src_points->points[s_index]);

                visz.at<cv::Vec3b>(j, i)[2] = 255;
             }
             // }
       }
    }
    
    target_points->clear();
    pcl::copyPointCloud<PointNormalT, PointNormalT>(
       *update_model, *target_points);
    pcl::PointCloud<PointNormalT>().swap(*update_model);
    
    cv::imshow("src_points", src_projection.rgb);
    cv::imshow("viz", visz);
    
    cv::waitKey(3);
}

void HandheldObjectRegistration::featureBasedTransformation(
    std::vector<CandidateIndices> &candidate_indices,
    const pcl::PointCloud<PointNormalT>::Ptr target_points,
    const cv::Mat target_indices, const cv::Mat target_image,
    const pcl::PointCloud<PointNormalT>::Ptr src_points,
    const cv::Mat src_indices, const cv::Mat src_image) {
    candidate_indices.clear();
    if (src_points->empty() || src_image.empty() || src_indices.empty() ||
        target_points->empty() || target_image.empty() ||
        target_indices.empty()) {
       ROS_ERROR("EMPTY INPUT FOR FEATUREBASEDTRANSFORM(.)");
       return;
    }
    
    std::vector<cv::KeyPoint> src_keypoints;
    cv::cuda::GpuMat d_src_desc;
    this->features2D(src_keypoints, d_src_desc, src_image);

    // TODO(CACHE): reuse
    cv::cuda::GpuMat d_tgt_desc;
    std::vector<cv::KeyPoint> tgt_keypoints;
    this->features2D(tgt_keypoints, d_tgt_desc, target_image);
    
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
       cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    // matcher->match(d_src_desc, d_tgt_desc, matches);

    std::vector<std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch(d_src_desc, d_tgt_desc, knn_matches, 2);

    const float ANGLE_THRESH_ = std::cos(M_PI / 3.0f);
    
    for (int i = 0; i < knn_matches.size(); i++) {
       if ((knn_matches[i][0].distance / knn_matches[i][1].distance) < 0.8f) {
          matches.push_back(knn_matches[i][0]);
       }
    }
    std::vector<bool> matches_flag(static_cast<int>(matches.size()));
    candidate_indices.resize(matches.size());
    
    // TODO(PARALLEL): OMP

    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < matches.size(); i++) {
       cv::Point2f src_pt = src_keypoints[matches[i].queryIdx].pt;
       cv::Point2f tgt_pt = tgt_keypoints[matches[i].trainIdx].pt;
       
       int src_index = src_indices.at<int>(static_cast<int>(src_pt.y),
                                           static_cast<int>(src_pt.x));
       int tgt_index = target_indices.at<int>(static_cast<int>(tgt_pt.y),
                                              static_cast<int>(tgt_pt.x));
       
       if (tgt_index != -1 && src_index != -1) {
          /*
          Eigen::Vector4f src_normal = src_points->points[
             src_index].getNormalVector4fMap();
          Eigen::Vector4f tgt_normal = target_points->points[
             tgt_index].getNormalVector4fMap();
          float angle = (src_normal.dot(tgt_normal)) / (
             src_normal.norm() * tgt_normal.norm());
          if (angle > ANGLE_THRESH_)
          */
          Eigen::Vector4f src_point = src_points->points[
             src_index].getVector4fMap();
          Eigen::Vector4f tgt_point = target_points->points[
             tgt_index].getVector4fMap();
          src_point(3) = 0.0f;
          tgt_point(3) = 0.0f;
          double d = pcl::distances::l2(src_point, tgt_point);
          if (d < 0.02) {
             matches_flag[i] = true;
             candidate_indices[i].source_index = src_index;
             candidate_indices[i].target_index = tgt_index;

             good_matches.push_back(matches[i]);
          }
       } else {
             matches_flag[i] = false;
             candidate_indices[i].source_index = -1;
             candidate_indices[i].target_index = -1;
       }
       // } //! endif
    }

    cv::Mat img_matches;
    cv::drawMatches(src_image, src_keypoints, target_image,  tgt_keypoints,
                    good_matches, img_matches);
    cv::imshow("matching", img_matches);
}


void HandheldObjectRegistration::features2D(
    std::vector<cv::KeyPoint> &keypoints,
    cv::cuda::GpuMat &d_descriptor, const cv::Mat image) {
    keypoints.clear();
    if (image.empty()) {
       return;
    }
    cv::cuda::GpuMat d_image(image);
    // cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(
    //    d_image.type(), -1, cv::Size(3, 3), 1);
    // gauss->apply(d_image, d_image);
    cv::cuda::cvtColor(d_image, d_image, CV_BGR2GRAY);
    orb_gpu_->detectAndCompute(d_image, cv::cuda::GpuMat(),
                              keypoints, d_descriptor, false);
}

bool HandheldObjectRegistration::registrationICP(
    pcl::PointCloud<PointNormalT>::Ptr align_points,
    Eigen::Matrix<float, 4, 4> &transformation,
    const pcl::PointCloud<PointNormalT>::Ptr src_points,
    const pcl::PointCloud<PointNormalT>::Ptr target_points) {
    if (src_points->empty() || target_points->empty()) {
       ROS_ERROR("- ICP FAILED. INCORRECT INPUT SIZE ");
       return false;
    }
    pcl::IterativeClosestPointWithNormals<PointNormalT, PointNormalT>::Ptr icp(
       new pcl::IterativeClosestPointWithNormals<PointNormalT, PointNormalT>);
    icp->setMaximumIterations(10);
    icp->setRANSACOutlierRejectionThreshold(0.05);
    icp->setRANSACIterations(1000);
    icp->setTransformationEpsilon(1e-8);
    icp->setUseReciprocalCorrespondences(true);
    icp->setMaxCorrespondenceDistance(0.03);
    icp->setInputSource(src_points);
    icp->setInputTarget(target_points);
    /*
    pcl::registration::TransformationEstimationSVD<
       PointNormalT, PointNormalT>::Ptr trans_svd(
          new pcl::registration::TransformationEstimationSVD<
          PointNormalT, PointNormalT>);
    */
    pcl::registration::TransformationEstimationPointToPlaneLLS<
       PointNormalT, PointNormalT>::Ptr trans_svd(
          new pcl::registration::TransformationEstimationPointToPlaneLLS<
          PointNormalT, PointNormalT>);
    icp->setTransformationEstimation(trans_svd);
    icp->align(*align_points);
    transformation = icp->getFinalTransformation();
    
    std::cout << "has converged:" << icp->hasConverged() << " score: "
              << icp->getFitnessScore() << std::endl;
    std::cout << "Rotation Threshold: " << icp->getTransformationEpsilon()  << "\n";
    
    return (icp->hasConverged());
}

float HandheldObjectRegistration::checkRegistrationFitness(
    const ProjectionMap src_projection,
    const pcl::PointCloud<PointNormalT>::Ptr src_points,
    const ProjectionMap target_projection,
    const pcl::PointCloud<PointNormalT>::Ptr target_points) {
    if (src_points->empty() || target_points->empty()) {
       ROS_ERROR("EMPTY INPUT.. REGISTRATION CHECK FAILED");
       return -1.0f;
    }
    double DIST_THRESH = 0.05;
    float inlier_counter = 0.0f;
    float outlier_counter = 0.0f;

    // cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
    for (int j = 0; j < src_projection.indices.rows; j++) {
       for (int i = 0; i < src_projection.indices.cols; i++) {
          int s_index = src_projection.indices.at<int>(j, i);
          int t_index = target_projection.indices.at<int>(j, i);
          // cv::Scalar color(255, 0, 0);
          if (s_index != -1 && t_index != -1) {
             float diff = std::fabs(src_points->points[s_index].z -
                                    target_points->points[t_index].z);
             if (diff < DIST_THRESH) {
                inlier_counter += 1.0f;
                // color = cv::Scalar(0, 255, 0);
             } else {
                outlier_counter += 1.0f;
             }
          } else if (s_index == -1 && t_index != -1) {
             outlier_counter += 1.0f;
          } else if (s_index != -1 && t_index == -1) {
             outlier_counter += 1.0f;
          }
          // cv::circle(image, cv::Point(i, j), 2, color, -1);
       }
    }
    // cv::imshow("render", image);
    // cv::waitKey(3);
    
    float fitness = inlier_counter / (inlier_counter + outlier_counter);
    return fitness;
}


void HandheldObjectRegistration::symmetricPlane(
    float *equation, pcl::PointCloud<PointNormalT>::Ptr in_cloud,
    const float leaf_size) {
    if (in_cloud->empty()) {
       ROS_ERROR("Input size are not equal");
       return;
    }
    pcl::PointCloud<PointNormalT>::Ptr region_points(
       new pcl::PointCloud<PointNormalT>);
    pcl::VoxelGrid<PointNormalT> voxel_grid;
    voxel_grid.setInputCloud(in_cloud);
    voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_grid.filter(*region_points);

    // in_cloud->clear();
    // *in_cloud = *region_points;

    pcl::KdTreeFLANN<PointNormalT>::Ptr kdtree(
       new pcl::KdTreeFLANN<PointNormalT>);
    kdtree->setInputCloud(region_points);

    const float ANGLE_THRESH_ = M_PI/8;
    const float DISTANCE_THRESH_ = 0.10f;
    
    //! compute the symmetric
    std::vector<std::vector<Eigen::Vector4f> > symmetric_planes;
    float symmetric_energy = 0.0f;
    for (int i = 0; i < region_points->size(); i++) {
       Eigen::Vector4f point1 = region_points->points[i].getVector4fMap();
       Eigen::Vector4f norm1 = region_points->points[i].getNormalVector4fMap();
       point1(3) = 0.0f;
       norm1(3) = 0.0f;
       
       std::vector<Eigen::Vector4f> s_planes;
       Eigen::Vector4f plane_param;
       for (int j = i + 1; j < region_points->size(); j++) {
          Eigen::Vector4f point2 = region_points->points[j].getVector4fMap();
          Eigen::Vector4f norm2 = region_points->points[
             j].getNormalVector4fMap();
          point2(3) = 0.0f;
          norm2(3) = 0.0f;

          //! equation of the symmetric plane
          double d = pcl::distances::l2(point1, point2);
          Eigen::Vector4f norm_s = (point1 - point2) / static_cast<float>(d);
          norm_s = norm_s.normalized();
          float dist_s = ((point1 - point2).dot(norm_s)) / 2.0f;
          norm_s(3) = dist_s;
          
          Eigen::Vector3f plane_n = norm_s.head<3>();
          //! compute fitness
          std::vector<int> neigbor_indices;
          std::vector<float> point_squared_distance;
          float weight = 0.0f;
          
          pcl::PointCloud<PointNormalT>::Ptr temp_points(
             new pcl::PointCloud<PointNormalT>);
          for (int k = 0; k < region_points->size(); k++) {
             Eigen::Vector3f n = region_points->points[
                k].getNormalVector3fMap();
             Eigen::Vector3f p = region_points->points[k].getVector3fMap();
                
             // float beta = p(0) * plane_n(0) + p(1) * plane_n(1) +
             //    p(2) * plane_n(2);
             // float alpha = (plane_n(0) * plane_n(0)) +
             //    (plane_n(1) * plane_n(1)) + (plane_n(2) * plane_n(2));
             // float t = (norm_s(3) - beta) / alpha;
             
             PointNormalT seed_point = region_points->points[k];
             // seed_point.x = p(0) + (t * 2 * plane_n(0));
             // seed_point.y = p(1) + (t * 2 * plane_n(1));
             // seed_point.z = p(2) + (t * 2 * plane_n(2));
             
             Eigen::Vector3f seed_pt = p - 2.0f * norm_s.head<3>() *
                (p.dot(norm_s.head<3>()) - d);
             seed_point.x = seed_pt(0);
             seed_point.y = seed_pt(1);
             seed_point.z = seed_pt(2);
                
             neigbor_indices.clear();
             point_squared_distance.clear();
             int search_out = kdtree->nearestKSearch(
                seed_point, 1, neigbor_indices, point_squared_distance);
             
             int nidx = neigbor_indices[0];
             float distance = std::sqrt(point_squared_distance[0]);
             
             if (distance < leaf_size * 2.0f) {
                Eigen::Vector3f symm_n = n - (2.0f * (
                                                 plane_n.dot(n)) * plane_n);
                Eigen::Vector3f sneig_r = region_points->points[
                      nidx].getNormalVector3fMap();
                float dot_prod = (symm_n.dot(sneig_r)) / (
                   symm_n.norm() * sneig_r.norm());
                float angle = std::acos(dot_prod);  //! * (180.0f/M_PI);
                float w = 1.0f / (1.0f + (angle / (2.0f * M_PI)));

                // std::cout << w  << "\t" << angle * 180.0/M_PI << "\n";
                
                if (angle > ANGLE_THRESH_) {
                   weight += (w * 0.5f);
                } else {
                   weight += w;
                }
             }
             temp_points->push_back(seed_point);
          }
          weight /= static_cast<float>(region_points->size());
          
          // printf("WEIGHT: %3.2f, %3.2f \n", weight, symmetric_energy);
          if (!isnan(weight) && weight > symmetric_energy) {
             symmetric_energy = weight;
             plane_param = norm_s;
             in_cloud->clear();
             *in_cloud = *temp_points;
          }
       }

       this->plotPlane(in_cloud, plane_param);
       return;
       
       if (!s_planes.empty()) {
          symmetric_planes.push_back(s_planes);
       }
       std::cout << "\033[35m NUMBER OF SYMMETRIC PLANES:  \033[0m"
                 << s_planes.size()  << "\n";

       //! cluster similar planes
       const float dist_thresh = 0.05f;
       std::vector<int> filter_plane_indices;
       for (int j = 1; j < s_planes.size(); j++) {
          float distance = std::fabs(s_planes[j-1](3) - s_planes[j](3));
          float angle = s_planes[j].head<3>().dot(s_planes[j-1].head<3>());

          std::cout << "\t\t: " << distance << "\t" << angle  << "\n";
          std::cout << s_planes[j] << "\n----\n" << s_planes[j-1]  << "\n";
          
          if (distance < dist_thresh && angle > 0.85f) {
             
          }
       }
       std::exit(-1);
    }
    
    std::cout << "\033[34m NUMBER OF SYMMETRIC PLANES:  \033[0m"
              << symmetric_planes.size()  << "\n";


    return;
}

bool HandheldObjectRegistration::seedRegionGrowing(
    pcl::PointCloud<PointNormalT>::Ptr src_points,
    const PointT seed_point, const PointCloud::Ptr cloud,
    PointNormal::Ptr normals) {
    if (cloud->empty() || normals->size() != cloud->size()) {
       ROS_ERROR("- Region growing failed. Incorrect inputs sizes ");
       return false;
    }
    if (isnan(seed_point.x) || isnan(seed_point.y) || isnan(seed_point.z)) {
       ROS_ERROR("- Seed Point is Nan. Skipping");
       return false;
    }
    
    std::vector<int> neigbor_indices;
    this->getPointNeigbour<int>(neigbor_indices, seed_point, 1);
    int seed_index = neigbor_indices[0];
    
    const int in_dim = static_cast<int>(cloud->size());
    int *labels = reinterpret_cast<int*>(malloc(sizeof(int) * in_dim));
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_)
#endif
    for (int i = 0; i < in_dim; i++) {
       if (i == seed_index) {
          labels[i] = 1;
       }
       labels[i] = -1;
    }
    this->seedCorrespondingRegion(labels, cloud, normals,
                                  seed_index, seed_index);
    src_points->clear();
    for (int i = 0; i < in_dim; i++) {
       if (labels[i] != -1) {
          PointNormalT pt;
          pt.x = cloud->points[i].x;
          pt.y = cloud->points[i].y;
          pt.z = cloud->points[i].z;
          pt.r = cloud->points[i].r;
          pt.g = cloud->points[i].g;
          pt.b = cloud->points[i].b;
          pt.normal_x = normals->points[i].normal_x;
          pt.normal_y = normals->points[i].normal_y;
          pt.normal_z = normals->points[i].normal_z;
          src_points->push_back(pt);
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
    this->getPointNeigbour<int>(neigbor_indices,
                                cloud->points[parent_index], 18);

    int neigb_lenght = static_cast<int>(neigbor_indices.size());
    std::vector<int> merge_list(neigb_lenght);
    merge_list[0] = -1;
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_)
#endif
    for (int i = 1; i < neigbor_indices.size(); i++) {
        int index = neigbor_indices[i];
        if (index != parent_index && labels[index] == -1) {
            Eigen::Vector4f parent_pt = cloud->points[
                parent_index].getVector4fMap();
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
    // float im_relation = (n_centroid - c_centroid).dot(n_normal);
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
    std::vector<int> &neigbor_indices,
    const PointT seed_point, const T K, bool is_knn) {
    if (isnan(seed_point.x) ||
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


bool HandheldObjectRegistration::project3DTo2DDepth(
    ProjectionMap &projection_map,
    const pcl::PointCloud<PointNormalT>::Ptr cloud, const float max_distance) {
    if (cloud->empty()) {
       ROS_ERROR("- Empty cloud. Cannot project 3D to 2D depth.");
       return false;
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
    cv::Mat rotation_matrix = cv::Mat(3, 3, CV_32F, R);
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
    cv::Rodrigues(rotation_matrix, rvec);
    
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(object_points, rvec, translation_matrix,
                      camera_matrix, distortion_model, image_points);
    
    cv::Mat image = cv::Mat::zeros(camera_info_->height,
                                   camera_info_->width, CV_8UC3);
    cv::Mat depth_image = cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat indices = cv::Mat(image.size(), CV_32S);
    
    cv::Mat flag = cv::Mat(image.size(), CV_32S);
    for (int j = 0; j < image.rows; j++) {
       for (int i = 0; i < image.cols; i++) {
          indices.at<int>(j, i) = -1;
          flag.at<int>(j, i) = 0;
       }
    }

    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = 0;
    int max_y = 0;

    std::vector<bool> visibility_flag(static_cast<int>(image_points.size()));
    for (int i = 0; i < image_points.size(); i++) {
       int x = image_points[i].x;
       int y = image_points[i].y;
       if (!isnan(x) && !isnan(y) && (x >= 0 && x <= image.cols) &&
           (y >= 0 && y <= image.rows)) {
          //! occlusion points
          if (flag.at<int>(y, x) == 1) {
             int prev_ind = indices.at<int>(y, x);
             float prev_depth = cloud->points[prev_ind].z;
             float curr_depth = cloud->points[i].z;
             if (curr_depth < prev_depth && curr_depth > 0.0f &&
                 !isnan(curr_depth)) {
                // TODO(NORMAL CHECK):  if normal deviates more
                // reject
                image.at<cv::Vec3b>(y, x)[2] = cloud->points[i].r;
                image.at<cv::Vec3b>(y, x)[1] = cloud->points[i].g;
                image.at<cv::Vec3b>(y, x)[0] = cloud->points[i].b;

                depth_image.at<float>(y, x) = cloud->points[i].z / max_distance;
                indices.at<int>(y, x) = i;

                 visibility_flag[prev_ind] = false;
             } else {
                visibility_flag[i] = false;
             }
          } else {
             flag.at<int>(y, x) = 1;
             image.at<cv::Vec3b>(y, x)[2] = cloud->points[i].r;
             image.at<cv::Vec3b>(y, x)[1] = cloud->points[i].g;
             image.at<cv::Vec3b>(y, x)[0] = cloud->points[i].b;
             
             depth_image.at<float>(y, x) = cloud->points[i].z / max_distance;
             indices.at<int>(y, x) = i;

             visibility_flag[i] = true;
          }
          min_x = (x < min_x) ? x : min_x;
          min_y = (y < min_y) ? y : min_y;
          max_x = (x > max_x) ? x : max_x;
          max_y = (y > max_y) ? y : max_y;
       }
    }
    int width = max_x - min_x;
    int height = max_y - min_y;
    cv::Rect rect(min_x, min_y, width, height);
    
    this->conditionROI(min_x, min_y, width, height, image.size());
    projection_map.x = min_x;
    projection_map.y = min_y;
    projection_map.width = width;
    projection_map.height = height;
    projection_map.rgb = image;
    projection_map.depth = depth_image;
    projection_map.indices = indices;
    projection_map.visibility_flag.clear();
    projection_map.visibility_flag = visibility_flag;
    
    return true;
}


bool HandheldObjectRegistration::conditionROI(
    int x, int y, int width, int height, const cv::Size image_size) {
    if (x > image_size.width || y > image_size.height) {
       return false;
    }
    
    x = (x < 0) ? 0 : x;
    y = (y < 0) ? 0 : y;
    x = (x > image_size.width) ? image_size.width : x;
    y = (y > image_size.height) ? image_size.height : y;
    width = (width + x > image_size.width) ?
       (width - ((width + x) - image_size.width)) : width;
    height = (height + y > image_size.height) ?
       (height - ((height + y) - image_size.height)) : height;

    return true;
}

void HandheldObjectRegistration::getObjectRegion(
    pcl::PointCloud<PointNormalT>::Ptr src_points,
    const PointCloud::Ptr cloud, const PointNormal::Ptr normals,
    const PointT seed_pt) {
    if (cloud->empty() || normals->size() != cloud->size()) {
       return;
    }

    cv::Point2f image_index;
    int seed_index = -1;
    if (this->projectPoint3DTo2DIndex(image_index, seed_pt)) {
       seed_index = (image_index.x + (image_index.y * camera_info_->width));
    } else {
       ROS_ERROR("INDEX IS NAN");
       return;
    }
    
    Eigen::Vector4f seed_point = cloud->points[seed_index].getVector4fMap();
    Eigen::Vector4f seed_normal = normals->points[
       seed_index].getNormalVector4fMap();
    
    std::vector<int> processing_list;
    std::vector<int> labels(static_cast<int>(cloud->size()), -1);

    const int window_size = 3;
    const int wsize = window_size * window_size;
    const int lenght = std::floor(window_size/3);

    processing_list.clear();
    for (int j = -lenght; j <= lenght; j++) {
       for (int i = -lenght; i <= lenght; i++) {
          int index = (seed_index + (j * camera_info_->width)) + i;
          if (index >= 0 && index < cloud->size()) {
             processing_list.push_back(index);
          }
       }
    }
    
    std::vector<int> temp_list;
    while (true) {
       
       if (processing_list.empty()) {
          break;
       }
       temp_list.clear();
       for (int i = 0; i < processing_list.size(); i++) {
          int idx = processing_list[i];
          if (labels[idx] == -1) {
             Eigen::Vector4f c = cloud->points[idx].getVector4fMap();
             Eigen::Vector4f n = normals->points[idx].getNormalVector4fMap();
             
             if (this->seedVoxelConvexityCriteria(
                    seed_point, seed_normal, seed_point, c, n, -0.01) == 1) {
                labels[idx] = 1;

                for (int j = -lenght; j <= lenght; j++) {
                   for (int k = -lenght; k <= lenght; k++) {
                      int index = (idx + (j * camera_info_->width)) + k;
                      if (index >= 0 && index < cloud->size()) {
                         temp_list.push_back(index);
                      }
                   }
                }
             }
          }
       }

       processing_list.clear();
       processing_list.insert(processing_list.end(), temp_list.begin(),
                              temp_list.end());
    }

    // PointCloud::Ptr points(new PointCloud);
    // for (int i = 0; i < labels.size(); i++) {
    //    if (labels[i] != -1) {
    //       points->push_back(cloud->points[i]);
    //    }
    // }
    src_points->clear();
    for (int i = 0; i < labels.size(); i++) {
       if (labels[i] != -1) {
          PointNormalT pt;
          pt.x = cloud->points[i].x;
          pt.y = cloud->points[i].y;
          pt.z = cloud->points[i].z;
          pt.r = cloud->points[i].r;
          pt.g = cloud->points[i].g;
          pt.b = cloud->points[i].b;
          pt.normal_x = normals->points[i].normal_x;
          pt.normal_y = normals->points[i].normal_y;
          pt.normal_z = normals->points[i].normal_z;
          src_points->push_back(pt);
       }
    }
    // cloud->clear();
    // *cloud = *points;
}

bool HandheldObjectRegistration::projectPoint3DTo2DIndex(
    cv::Point2f &image_index, const PointT in_point) {
    if (isnan(in_point.x) || isnan(in_point.y) || isnan(in_point.z)) {
       ROS_ERROR(" NAN POINT CANNOT BE PROJECTED TO 2D INDEX.");
       return false;
    }
    cv::Mat object_points = cv::Mat(1, 3, CV_32F);
    object_points.at<float>(0, 0) = in_point.x;
    object_points.at<float>(0, 1) = in_point.y;
    object_points.at<float>(0, 2) = in_point.z;

    float K[9];
    float R[9];
    for (int i = 0; i < 9; i++) {
       K[i] = camera_info_->K[i];
       R[i] = camera_info_->R[i];
    }

    cv::Mat camera_matrix = cv::Mat(3, 3, CV_32F, K);
    cv::Mat rotation_matrix = cv::Mat(3, 3, CV_32F, R);
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
    cv::Rodrigues(rotation_matrix, rvec);
    
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(object_points, rvec, translation_matrix,
                      camera_matrix, distortion_model, image_points);
    if ((image_points[0].x >= 0 && image_points[0].x < camera_info_->width) &&
        (image_points[0].y >= 0 && image_points[0].y < camera_info_->height)) {
       image_index = image_points[0];
       return true;
    } else {
       return false;
    }
}


void HandheldObjectRegistration::getAxisAngles(
    float &angle_x, float &angle_y, float &angle_z,
    const Eigen::Matrix4f transform_matrix) {
    angle_x = std::atan2(transform_matrix(2, 1),
                               transform_matrix(2, 2));
    angle_y = std::atan2(-transform_matrix(2, 0), std::sqrt(
          transform_matrix(2, 1) * transform_matrix(2, 1) +
          transform_matrix(2, 2) * transform_matrix(2, 2)));
    angle_z = std::atan2(transform_matrix(1, 0), transform_matrix(0, 0));
}


/**
 * DEGUB FUNCTIONS ONLY
 */
void HandheldObjectRegistration::plotPlane(
    pcl::PointCloud<PointNormalT>::Ptr cloud, const Eigen::Vector4f param,
    const Eigen::Vector3f color) {
    Eigen::Vector3f center = Eigen::Vector3f(param(3)/param(0), 0, 0);
    // Eigen::Vector3f normal = param.head<3>();
    // float coef = normal.dot(center);
    // float x = coef / normal(0);
    // float y = coef / normal(1);
    // float z = coef / normal(2);
    Eigen::Vector3f point_x = Eigen::Vector3f(param(3)/param(0), 0.0f, 0.0f);
    Eigen::Vector3f point_y = Eigen::Vector3f(-param(2), 0, param(0));
    Eigen::Vector3f point_z = Eigen::Vector3f(-param(1), param(0), 0);
    for (float y = -1.0f; y < 1.0f; y += 0.01f) {
       for (float x = -1.0f; x < 1.0f; x += 0.01f) {
         PointNormalT pt;
         pt.x = point_x(0) + point_y(0) * x + point_z(0) * y;
         pt.y = point_x(1) + point_y(1) * x + point_z(1) * y;
         pt.z = point_x(2) + point_y(2) * x + point_z(2) * y;
         pt.g = 255;
         cloud->push_back(pt);
      }
    }
}

void HandheldObjectRegistration::getPFTransformation() {
    tf::TransformListener tf_listener;
    tf::StampedTransform transform;
    ros::Time now = ros::Time(0);
    std::string child_frame = "/camera_rgb_optical_frame";
    std::string parent_frame = "/track_result";
    Eigen::Affine3f transform_model = Eigen::Affine3f::Identity();
    tf::Transform update_transform;
    std::vector<int> temp;
    
    bool wft_ok = tf_listener.waitForTransform(
       child_frame, parent_frame, now, ros::Duration(1));
    if (!wft_ok) {
       ROS_ERROR("CANNOT TRANSFORM SOURCE AND TARGET FRAMES");
       return;
    }
    tf_listener.lookupTransform(
       child_frame, parent_frame, now, transform);
    tf::Quaternion tf_quaternion =  transform.getRotation();
    transform_model = Eigen::Affine3f::Identity();
    transform_model.translation() <<
       transform.getOrigin().getX(),
       transform.getOrigin().getY(),
       transform.getOrigin().getZ();
    Eigen::Quaternion<float> quaternion = Eigen::Quaternion<float>(
       tf_quaternion.w(), tf_quaternion.x(),
       tf_quaternion.y(), tf_quaternion.z());
    transform_model.rotate(quaternion);
    tf::Vector3 origin = tf::Vector3(transform.getOrigin().getX(),
                                     transform.getOrigin().getY(),
                                     transform.getOrigin().getZ());
    update_transform.setOrigin(origin);
    tf::Quaternion update_quaternion = tf_quaternion;

    update_transform.setRotation(update_quaternion +
                                 this->previous_transform_.getRotation());

    static tf::TransformBroadcaster br;
    br.sendTransform(tf::StampedTransform(
                        update_transform, camera_info_->header.stamp,
                        camera_info_->header.frame_id, "object_pose"));

    this->previous_transform_ = update_transform;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "handheld_object_registration");
    HandheldObjectRegistration hor;
    ros::spin();
    return 0;
}

