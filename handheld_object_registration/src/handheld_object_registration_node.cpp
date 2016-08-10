
#include <handheld_object_registration/handheld_object_registration.h>

HandheldObjectRegistration::HandheldObjectRegistration():
    num_threads_(16), is_init_(false), min_points_size_(100),
    weight_decay_factor_(0.6f), init_weight_(1.0f), pose_flag_(false) {
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    this->target_points_ = pcl::PointCloud<PointNormalT>::Ptr(
       new pcl::PointCloud<PointNormalT>);
    this->prev_points_ = pcl::PointCloud<PointNormalT>::Ptr(
       new pcl::PointCloud<PointNormalT>);
    this->input_cloud_ = PointCloud::Ptr(new PointCloud);
    this->input_normals_ = PointNormal::Ptr(new PointNormal);
    
    this->orb_gpu_ = cv::cuda::ORB::create(500, 1.10f, 4, 31, 0, 2,
                                           cv::ORB::HARRIS_SCORE, 20);
     
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
    
    this->camera_info_ = cinfo_msg;
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    this->kdtree_->setInputCloud(cloud);

    PointNormal::Ptr normals(new PointNormal);
    this->getNormals(normals, cloud);

    this->input_cloud_->clear();
    pcl::copyPointCloud<PointT, PointT>(*cloud, *input_cloud_);
    this->input_normals_->clear();
    pcl::copyPointCloud<NormalT, NormalT>(*normals, *input_normals_);
    
    PointCloud::Ptr region_cloud(new PointCloud);
    PointNormal::Ptr region_normal(new PointNormal);

    int seed_index  = screen_msg_.point.x + (640 * screen_msg_.point.y);
    PointT seed_point = cloud->points[seed_index];
    if (this->pose_flag_) {
       seed_point.x = pose_msg_->pose.position.x;
       seed_point.y = pose_msg_->pose.position.y;
       seed_point.z = pose_msg_->pose.position.z;
       
       std::cout << seed_point  << "\n";
       std::cout << pose_msg_->pose.position.x << ", " <<
          pose_msg_->pose.position.y << ", " <<
          pose_msg_->pose.position.z << "\n";

    }
    
    this->seedRegionGrowing(region_cloud, region_normal, seed_point,
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
       transform_model.translation() << pose_msg->pose.position.x,
          pose_msg->pose.position.y, pose_msg->pose.position.z;
       Eigen::Quaternion<float> pf_quat = Eigen::Quaternion<float>(
          pose_msg->pose.orientation.w, pose_msg->pose.orientation.x,
          pose_msg->pose.orientation.y, pose_msg->pose.orientation.z);
       transform_model.rotate(pf_quat);
       if (!this->prev_points_->empty()) {
          tracker_transform = transform_model * prev_transform_.inverse();
          pcl::transformPointCloud(*prev_points_,
                                   *prev_points_, tracker_transform);
          this->pose_flag_ = false;
       }
       prev_transform_ = transform_model;
    } else {
       ROS_WARN("NOT SET");
    }
    /**
     * END
     */
    
    
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

       transformation =  transformation * tracker_transform.matrix();
       // pcl::transformPointCloud(*target_points_, *target_points_,
       //                          transformation);
       
       /*
       pcl::PointCloud<PointNormalT>::Ptr tmp_cloud(
          new pcl::PointCloud<PointNormalT>);
       PointCloud::Ptr tmp_cloud (new PointCloud);
       pcl::transformPointCloud(*src_points, *tmp_cloud, transformation);
       */
       
       this->modelUpdate(src_points, target_points_, transformation);
       
       sensor_msgs::PointCloud2 ros_cloud;
       pcl::toROSMsg(*src_points, ros_cloud);
       // pcl::toROSMsg(*align_points, ros_cloud);
       ros_cloud.header = cloud_msg->header;
       this->pub_icp_.publish(ros_cloud);
    } else {
       this->target_points_->clear();
       *target_points_ = *src_points;
    }

    this->prev_points_->clear();
    // pcl::copyPointCloud<PointNormalT, PointNormalT>(*src_points,
    // *prev_points_);
    pcl::copyPointCloud<PointNormalT, PointNormalT>(
       *target_points_, *prev_points_);
    
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
    
    cv::Mat src_image;
    cv::Mat src_depth;
    cv::Mat src_indices = this->project3DTo2DDepth(
       src_image, src_depth, src_points/*trans_points*/);
    
    //! move it out***
    cv::Mat target_image;
    cv::Mat target_depth;
    cv::Mat target_indices = this->project3DTo2DDepth(
       target_image, target_depth, target_points);

    //!  Feature check
    std::vector<CandidateIndices> candidate_indices;
    this->featureBasedTransformation(
       candidate_indices, target_points, target_indices, target_image,
       src_points, src_indices, src_image);

    //! compute transformation
    Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity();
    if (candidate_indices.size() < 8) {
       //! use only ICP
    } else {
       float energy = FLT_MAX;
       Eigen::Matrix3f rotation;
       Eigen::Vector3f transl;

       PointCloud::Ptr src_cloud(new PointCloud);
       PointCloud::Ptr target_cloud(new PointCloud);
       
       for (int i = 0; i < candidate_indices.size(); i++) {
          int src_index = candidate_indices[i].source_index;
          int tgt_index = candidate_indices[i].target_index;
          if (src_index != -1 && tgt_index != -1) {
             PointNormalT spt = src_points->points[src_index];
             PointNormalT tpt = target_points->points[tgt_index];
             PointT pt;
             pt.x = spt.x; pt.y = spt.y; pt.z = spt.z;
             src_cloud->push_back(pt);
             pt.x = tpt.x; pt.y = tpt.y; pt.z = tpt.z;
             target_cloud->push_back(pt);
          }
       }

       if (!src_cloud->empty() || !target_cloud->empty()) {
          pcl::registration::TransformationEstimationSVD<
             PointT, PointT> transformation_estimation;
          transformation_estimation.estimateRigidTransformation(
             *target_cloud, *src_cloud, transform_matrix);
          
          std::cout << transform_matrix  << "\n";
       }
       
       pcl::transformPointCloud(*src_points, *src_points, transform_matrix);

    }
    
    
    cv::waitKey(3);
    return;


    
    cv::Mat depth_im = cv::Mat::zeros(src_depth.size(), CV_32F);
    // cv::absdiff(src_depth, target_depth, depth_im);

    pcl::PointCloud<PointNormalT>::Ptr update_model(
       new pcl::PointCloud<PointNormalT>);
    pcl::copyPointCloud<PointNormalT, PointNormalT>(
       *target_points, *update_model);
    
    for (int j = 0; j < src_depth.rows; j++) {
       for (int i = 0; i < src_depth.cols; i++) {
          float t_depth = target_depth.at<float>(j, i);
          float s_depth = src_depth.at<float>(j, i);
          int s_index = src_indices.at<int>(j, i);
          int t_index = target_indices.at<int>(j, i);
          
          if ((t_index < static_cast<int>(target_points->size()) &&
               t_index != -1) || (s_index < static_cast<int>(src_points->size())
                                  && s_index != -1)) {
             
             if (s_index == -1 && t_index != -1) {
                //! keep this point
                // update_model->push_back(target_points->points[t_index]);
             
             } else if (s_index != -1 && t_index != -1) {
                //! replace target with src
                update_model->points[t_index] = src_points->points[s_index];
             } else if (s_index != -1 && t_index == -1) {
                //! probably new point
                update_model->push_back(src_points->points[s_index]);
             }
             
          
             float diff = src_depth.at<float>(j, i) -
                target_depth.at<float>(j, i);

             // std::cout << "diff: " << diff << "\t"
             //           << src_depth.at<float>(j, i) << "\t"
             //           << target_depth.at<float>(j, i) << "\n";
          
             if (std::fabs(diff) > 0.01f)  {
                depth_im.at<float>(j, i) = 1.0f;
             }
             
          }
          // else {
          //    ROS_ERROR("- index out of size");
          //    std::cout << s_index << "\t" << src_points->size()  << "\n";
          //    std::cout << t_index << "\t" << target_points->size()  << "\n";
          // }
          
       }
    }

    target_points->clear();
    pcl::copyPointCloud<PointNormalT, PointNormalT>(
       *update_model, *target_points);
    pcl::PointCloud<PointNormalT>().swap(*update_model);
    
    cv::imshow("src_points", src_image);
    cv::imshow("target_points", target_image);
    cv::imshow("depth", depth_im);
    cv::waitKey(3);

    return;
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
    
    cv::cuda::GpuMat d_tgt_desc;
    std::vector<cv::KeyPoint> tgt_keypoints;
    this->features2D(tgt_keypoints, d_tgt_desc, target_image);
    
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
       cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher->match(d_src_desc, d_tgt_desc, matches);

    const float ANGLE_THRESH_ = M_PI / 6.0f;
    std::vector<bool> matches_flag(static_cast<int>(matches.size()));
    candidate_indices.resize(matches.size());

    // TODO(PARALLEL): OMP
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < matches.size(); i++) {
       cv::Point2f src_pt = src_keypoints[matches[i].queryIdx].pt;
       cv::Point2f tgt_pt = tgt_keypoints[matches[i].trainIdx].pt;
       int src_index = src_indices.at<int>(static_cast<int>(tgt_pt.y),
                                           static_cast<int>(tgt_pt.x));
       int tgt_index = target_indices.at<int>(static_cast<int>(tgt_pt.y),
                                              static_cast<int>(tgt_pt.x));
       // std::cout << tgt_index  << "\t" << src_index << "\n";

       if (tgt_index != -1 && src_index != -1) {
          Eigen::Vector4f src_normal = src_points->points[
             src_index].getNormalVector4fMap();
          Eigen::Vector4f tgt_normal = target_points->points[
             tgt_index].getNormalVector4fMap();

          float angle = std::acos((src_normal.dot(tgt_normal)) / (
                                     src_normal.norm() * tgt_normal.norm()));
          if (angle < ANGLE_THRESH_) {
             matches_flag[i] = true;
             candidate_indices[i].source_index = src_index;
             candidate_indices[i].target_index = tgt_index;

             good_matches.push_back(matches[i]);
          } else {
             matches_flag[i] = false;
             candidate_indices[i].source_index = -1;
             candidate_indices[i].target_index = -1;
          }
       }
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
    cv::cuda::cvtColor(d_image, d_image, CV_BGR2GRAY);
    orb_gpu_->detectAndCompute(d_image, cv::cuda::GpuMat(),
                              keypoints, d_descriptor, false);
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
    icp->setRANSACOutlierRejectionThreshold(0.005);
    icp->setRANSACIterations(1000);
    
    // icp->setMaxCorrespondenceDistance(1e-5);
       
    // icp->setInputSource(src_points);
    icp->setInputSource(this->prev_points_);
    // icp->setInputTarget(this->target_points_);
    icp->setInputTarget(src_points);

    //! getting the correspondances
    pcl::registration::CorrespondenceEstimation<PointNormalT,
                                                PointNormalT>::Ptr
       correspond_estimation(new pcl::registration::CorrespondenceEstimation<
                             PointNormalT, PointNormalT>);
    // correspond_estimation->setInputSource(src_points);
    correspond_estimation->setInputSource(this->prev_points_);
    // correspond_estimation->setInputTarget(this->target_points_);
    correspond_estimation->setInputTarget(src_points);

    icp->setCorrespondenceEstimation(correspond_estimation);
    icp->align(*align_points);
    transformation = icp->getFinalTransformation();
    
    boost::shared_ptr<pcl::Correspondences> correspondences(
       new pcl::Correspondences);
    correspond_estimation->determineCorrespondences(*correspondences);

    ROS_INFO("PRINTING CORRESPONDENCES");

    align_points->clear();
    *align_points = *src_points;
    float fitness = icp->getFitnessScore();


    std::cout << "\033[33m INFO:  \033[0m" << correspondences->size()
              << "\t"  << align_points->size()  << "\n";

    float outlier_counter = 0.0f;
    for (int i = 0; i < correspondences->size(); i++) {
       int query = correspondences->operator[](i).index_query;  //! source
       int match = correspondences->operator[](i).index_match;  //! target
       if (correspondences->operator[](i).distance > 0.005) {

          // std::cout << "\033[35mDEBUG:  \033[0m" << query << ", "
          //           << match << "\t" << distance  << "\n";
          
          // this->target_points_->push_back(align_points->points[query]);
          // align_points->points[query].r = 0;
          // align_points->points[query].b = 0;
          // align_points->points[query].g = 0;

          outlier_counter += 1.0f;
       }
    }

    const float OUTLIER_THRESH_ = 0.05f;
    bool is_converged = (icp->hasConverged() == 1) ? true : false;
    float outlier_ratio = outlier_counter / static_cast<float>(
       correspondences->size());
    if (outlier_ratio > OUTLIER_THRESH_ && is_converged) {
       is_converged = false;
    }
    
    std::cout << "\033[34m OUTLIER RATION: " <<
       outlier_ratio << "\033[0m\n";
    
    std::cout << "has converged:" << icp->hasConverged() << " score: "
              << icp->getFitnessScore() << std::endl;
    std::cout << "\033[32mMax correspondence: " <<
       icp ->getMaxCorrespondenceDistance() << "\033[0m\n";
    
    return is_converged;
}

bool HandheldObjectRegistration::checkRegistrationFailure(
    const pcl::PointCloud<PointNormalT>::Ptr aligned_points,
    const pcl::PointCloud<PointNormalT>::Ptr prev_points) {
    if (aligned_points->empty() || prev_points->empty()) {
       ROS_ERROR("EMPTY INPUT.. REGISTRATION CHECK FAILED");
       return false;
    }

    
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
    PointCloud::Ptr out_cloud, PointNormal::Ptr out_normals,
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
    this->getPointNeigbour<int>(neigbor_indices,
                                cloud->points[parent_index], 18);

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

    cv::Mat flag = cv::Mat(image.size(), CV_32S);
    
    for (int j = 0; j < image.rows; j++) {
       for (int i = 0; i < image.cols; i++) {
          indices.at<int>(j, i) = -1;

          flag.at<int>(j, i) = 0;
       }
    }
    
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(this->num_threads_)
// #endif
    
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
             }
          } else {
             flag.at<int>(y, x) = 1;
             image.at<cv::Vec3b>(y, x)[2] = cloud->points[i].r;
             image.at<cv::Vec3b>(y, x)[1] = cloud->points[i].g;
             image.at<cv::Vec3b>(y, x)[0] = cloud->points[i].b;
             
             depth_image.at<float>(y, x) = cloud->points[i].z / max_distance;
             indices.at<int>(y, x) = i;
          }
       }
    }
    
    return indices;
}


/**
 * DEGUB ONE
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

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "handheld_object_registration");
    HandheldObjectRegistration hor;
    ros::spin();
    return 0;
}

