// Copyright (C) 2016 by Krishneel Chaudhary, JSK Lab,a
// The University of Tokyo, Japan

#include <interactive_segmentation/object_region_estimation.h>

ObjectRegionEstimation::ObjectRegionEstimation() :
    num_threads_(8), is_prev_ok_(false), min_cluster_size_(100) {
    this->go_signal_ = false;
    
    this->prev_cloud_ = pcl::PointCloud<PointT>::Ptr(
       new pcl::PointCloud<PointT>);
    this->onInit();
}

void ObjectRegionEstimation::onInit() {

    this->srv_client_ = this->pnh_.serviceClient<
       interactive_segmentation::Feature3DClustering>(
          "feature3d_clustering_srv");
  
    this->subscribe();
    
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/object_region_estimation/output/cloud", 1);
    
    this->pub_indices_ = this->pnh_.advertise<
       jsk_recognition_msgs::ClusterPointIndices>(
          "/object_region_estimation/output/indices", 1);

    this->pub_flow_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/object_region_estimation/output/scene_flow", 1);

    this->pub_signal_ = this->pnh_.advertise<Int32Stamped>(
       "/object_region_estimation/failure/signal", 1);
}

void ObjectRegionEstimation::subscribe() {

    // info before robot pushes the object
    this->sub_cloud_prev_.subscribe(this->pnh_, "prev_cloud", 1);
    this->sub_image_prev_.subscribe(this->pnh_, "prev_image", 1);
    this->sub_plane_prev_.subscribe(this->pnh_, "prev_plane", 1);
    this->sync_prev_ = boost::make_shared<message_filters::Synchronizer<
      SyncPolicyPrev> >(100);
    this->sync_prev_->connectInput(
       sub_image_prev_, sub_cloud_prev_, sub_plane_prev_);
    this->sync_prev_->registerCallback(
       boost::bind(&ObjectRegionEstimation::callbackPrev, this, _1, _2, _3));

    
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_image_.subscribe(this->pnh_, "input_image", 1);
    this->sub_original_.subscribe(this->pnh_, "input_orig", 1);
    this->sub_pose_.subscribe(this->pnh_, "input_pose", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(sub_image_, sub_cloud_, sub_original_, sub_pose_);
    this->sync_->registerCallback(
       boost::bind(&ObjectRegionEstimation::callback, this, _1, _2, _3, _4));
}

void ObjectRegionEstimation::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_image_.unsubscribe();
    this->sub_original_.unsubscribe();
    this->sub_cloud_prev_.unsubscribe();
    this->sub_image_prev_.unsubscribe();
}

void ObjectRegionEstimation::callbackPrev(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::PointCloud2::ConstPtr &plane_msg) {
    // if (is_prev_ok_) {
    //    return;
    // }
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(
          image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return;
    }
    if (cloud->empty() || cv_ptr->image.empty()) {
       is_prev_ok_ = false;
    } else {
       pcl::PointCloud<pcl::PointNormal>::Ptr plane_info(
          new pcl::PointCloud<pcl::PointNormal>);
       pcl::fromROSMsg(*plane_msg, *plane_info);
       this->plane_norm_ = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
       this->plane_point_ = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
       this->plane_norm_ = plane_info->points[0].getNormalVector3fMap();
       this->plane_point_ = plane_info->points[0].getVector3fMap();
       pcl::copyPointCloud<PointT, PointT>(*cloud, *prev_cloud_);
       this->prev_image_ = cv_ptr->image.clone();
       is_prev_ok_ = true;
    }
    ROS_INFO("SEGMENTED REGION SET...");
}

void ObjectRegionEstimation::callback(
    const sensor_msgs::Image::ConstPtr &image_msg,   // table -
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,  // table -
    const sensor_msgs::PointCloud2::ConstPtr &orig_msg,   // full scene
    const geometry_msgs::PoseStamped::ConstPtr &pose_msg) {  // end-eff
    if (!is_prev_ok_) {
       ROS_ERROR("ERROR: PREV INFO NOT FOUND");
       return;
    }
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr in_cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    pcl::fromROSMsg(*orig_msg, *in_cloud);
    this->header_ = cloud_msg->header;
    if (cloud->empty()) {
       ROS_ERROR("ERROR: EMPTY DATA.. SKIP MERGING");
       return;
    }
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(
          image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return;
    }
    cv::Mat next_img = cv_ptr->image.clone();
    pcl::PointCloud<PointT>::Ptr motion_region_cloud(
        new pcl::PointCloud<PointT>);
    this->sceneFlow(motion_region_cloud, this->prev_image_, next_img,
                    in_cloud, this->plane_norm_, this->plane_point_);

    ROS_INFO("SEGMENT CLUSTERING");
    
    // cluster regions into different segments
    std::vector<pcl::PointIndices> cluster_indices;
    this->clusterSegments(cluster_indices, motion_region_cloud);

    ROS_INFO("CLUSTERING DONE");
    
    // clarify each segments
    pcl::ExtractIndices<PointT>::Ptr eifilter(new pcl::ExtractIndices<PointT>);
    eifilter->setInputCloud(motion_region_cloud);
    cloud->clear();

    std::vector<pcl::PointIndices> all_indices;
    for (int i = 0; i < cluster_indices.size(); i++) {
       pcl::PointCloud<PointT>::Ptr region_cloud(
          new pcl::PointCloud<PointT>);
       pcl::PointIndices::Ptr region_indices(new pcl::PointIndices);
       *region_indices = cluster_indices[i];
       eifilter->setIndices(region_indices);
       eifilter->filter(*region_cloud);

       ROS_INFO("GETTING HYPOTHESIS");

       // TODO(fix): MIGHT HAVE BUG
       if (this->getHypothesis(this->prev_cloud_, region_cloud)) {
          pcl::copyPointCloud<PointT, PointT>(*region_cloud, *cloud);
          all_indices.clear();
          all_indices.push_back(cluster_indices[i]);
       }
    }
    
    // this->clusterFeatures(all_indices, cloud, normals, 5, 0.5);
    if (all_indices.empty()) {
       ROS_ERROR("\nOBJECT MERGING FAIL. RESETTING SEGMENTATION");
       Int32Stamped fail_signal;
       fail_signal.header = cloud_msg->header;
       fail_signal.data = -1;
       this->pub_signal_.publish(fail_signal);
    } else {
       jsk_recognition_msgs::ClusterPointIndices ros_indices;
       ros_indices.cluster_indices = pcl_conversions::convertToROSPointIndices(
          all_indices, cloud_msg->header);
       ros_indices.header = cloud_msg->header;
       pub_indices_.publish(ros_indices);
    
       sensor_msgs::PointCloud2 ros_cloud;
       pcl::toROSMsg(*cloud, ros_cloud);
       ros_cloud.header = cloud_msg->header;
       this->pub_cloud_.publish(ros_cloud);
    }
}

bool ObjectRegionEstimation::convertToCvMat(
    cv::Mat &image, const sensor_msgs::Image::ConstPtr &image_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(
          image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return false;
    }
    image = cv_ptr->image.clone();
    return true;
}

void ObjectRegionEstimation::sceneFlow(
    pcl::PointCloud<PointT>::Ptr filtered, const cv::Mat prev_image,
    const cv::Mat next_image, const pcl::PointCloud<PointT>::Ptr cloud,
    const Eigen::Vector3f plane_norm, const Eigen::Vector3f plane_pt) {
    cv::Mat prev_img;
    cv::cvtColor(prev_image, prev_img, CV_BGR2GRAY);
    cv::Mat next_img;
    cv::cvtColor(next_image, next_img, CV_BGR2GRAY);
    cv::Mat flow = cv::Mat(0, 0, CV_8UC1);
    cv::calcOpticalFlowFarneback(prev_img, next_img, flow, 0.2, 5, 7,
                                 10, 5, 1.2, 0);
    std::vector<cv::Mat> split_flow;
    cv::split(flow, split_flow);
    cv::Mat angle = cv::Mat::zeros(flow.size(), CV_32F);
    cv::Mat magnitude = cv::Mat::zeros(flow.size(), CV_32F);
    cv::cartToPolar(split_flow[0], split_flow[1], magnitude, angle, true);
    cv::Mat result;
    magnitude.convertTo(result, CV_8U);

    cv::Mat orientation = cv::Mat::zeros(angle.size(), CV_8UC3);
    cv::Mat image = cv::Mat::zeros(next_img.size(), next_img.type());

    // for the threshold use end-effector
    Eigen::Vector3f plane_point = plane_pt;
    plane_point(1) -= 0.02f;
    for (int j = 0; j < angle.rows; j++) {
       for (int i = 0; i < angle.cols; i++) {
          float val = angle.at<float>(j, i);
          if ((val > 0.0 && val < 360.0) && magnitude.at<float>(j, i) > 5.0f) {
             val /= 360.0f;
             orientation.at<cv::Vec3b>(j, i) [0]=  val * 255.0f;
             image.at<uchar>(j, i) += next_img.at<uchar>(j, i);
             PointT pt = cloud->points[i + (j * angle.cols)];
             if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z)) {
                if (plane_norm.dot(pt.getVector3fMap() - plane_point) >= 0.0f) {
                   filtered->push_back(pt);
                }
             }
          } else {
             orientation.at<cv::Vec3b>(j, i)[1] =  val * 255.0f;
          }
       }
    }
    cv::Mat flow_map;
    cv::cvtColor(next_img, flow_map, CV_GRAY2BGR);
    plotOpticalFlowMap(flow, flow_map, 5, 50, cv::Scalar(0, 255, 0));

    cv_bridge::CvImagePtr pub_flow(new cv_bridge::CvImage);
    pub_flow->header = header_;
    pub_flow->encoding = sensor_msgs::image_encodings::BGR8;
    pub_flow->image = flow_map.clone();
    this->pub_flow_.publish(pub_flow);
}

void ObjectRegionEstimation::plotOpticalFlowMap(
    const cv::Mat& flow, cv::Mat& flow_map, int step,
    double scale, const cv::Scalar& color) {
    for (int y = 0; y < flow_map.rows; y += step) {
      for (int x = 0; x < flow_map.cols; x += step) {
        const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
        cv::line(flow_map, cv::Point(x, y),
                 cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
        cv::circle(flow_map, cv::Point(cvRound(x+fxy.x),
                                       cvRound(y+fxy.y)), 1, color, -1);
      }
    }
}

void ObjectRegionEstimation::clusterSegments(
    std::vector<pcl::PointIndices> &cluster_indices,
    const pcl::PointCloud<PointT>::Ptr cloud) {
    if (cloud->empty()) {
      return;
    }
    cluster_indices.clear();
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);
    pcl::EuclideanClusterExtraction<PointT> euclidean_clustering;
    euclidean_clustering.setClusterTolerance(0.01f);
    euclidean_clustering.setMinClusterSize(this->min_cluster_size_);
    euclidean_clustering.setMaxClusterSize(25000);
    euclidean_clustering.setSearchMethod(tree);
    euclidean_clustering.setInputCloud(cloud);
    euclidean_clustering.extract(cluster_indices);

    /*
    int index = -1;
    int size = 0;
    for (int i = 0; i < cluster_indices.size(); i++) {
      if (cluster_indices[i].indices.size() > size) {
        size = cluster_indices[i].indices.size();
        index = i;
      }
    }
    pcl::PointCloud<PointT>::Ptr object_cloud(new pcl::PointCloud<PointT>);
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    pcl::ExtractIndices<PointT>::Ptr eifilter(new pcl::ExtractIndices<PointT>);
    eifilter->setInputCloud(cloud);
    *indices = cluster_indices[index];
    eifilter->setIndices(indices);
    eifilter->filter(*object_cloud);
    */
}

void ObjectRegionEstimation::keypoints3D(
    pcl::PointCloud<PointT>::Ptr keypoints,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const float radius_thresh, const bool uniform_keypoints) {
    if (cloud->empty()) {
       return;
    }
    keypoints->clear();
    if (uniform_keypoints) {
       pcl::PointCloud<int> keypoint_idx;
       pcl::UniformSampling<PointT> uniform_sampling;
       uniform_sampling.setInputCloud(cloud);
       uniform_sampling.setRadiusSearch(radius_thresh);
       uniform_sampling.compute(keypoint_idx);
       pcl::copyPointCloud<PointT, PointT>(
          *cloud, keypoint_idx.points, *keypoints);
    } else {
       pcl::HarrisKeypoint3D<PointT, PointI>::Ptr detector(
          new pcl::HarrisKeypoint3D<PointT, PointI>);
       detector->setNonMaxSupression(true);
       detector->setInputCloud(cloud);
       detector->setThreshold(1e-6);
       pcl::PointCloud<PointI>::Ptr keypoint_i(new pcl::PointCloud<PointI>);
       detector->compute(*keypoint_i);
       pcl::copyPointCloud<PointI, PointT>(*keypoint_i, *keypoints);
    }
}

void ObjectRegionEstimation::features3D(
    pcl::PointCloud<SHOT1344>::Ptr descriptors,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<Normal>::Ptr normals,
    const pcl::PointCloud<PointT>::Ptr keypoints,
    const float radius_thresh) {
    // pcl::SHOTEstimationOMP<PointT, Normal, SHOT352> shot;
    pcl::SHOTColorEstimationOMP<PointT, Normal, SHOT1344> shot;
    shot.setSearchSurface(cloud);
    shot.setInputCloud(keypoints);
    shot.setInputNormals(normals);
    shot.setRadiusSearch(radius_thresh);
    shot.compute(*descriptors);
}

bool ObjectRegionEstimation::getHypothesis(
    const pcl::PointCloud<PointT>::Ptr prev_cloud,
    const pcl::PointCloud<PointT>::Ptr next_cloud) {
    if (prev_cloud->empty() || next_cloud->empty()) {
      ROS_ERROR("ERROR: EMPTY CANNOT CREATE OBJECT HYPOTHESIS");
      return false;
    }
    const int k = 50;
    pcl::PointCloud<Normal>::Ptr prev_normals(new pcl::PointCloud<Normal>);
    pcl::PointCloud<Normal>::Ptr next_normals(new pcl::PointCloud<Normal>);
    pcl::PointCloud<PointT>::Ptr prev_keypoints(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr next_keypoints(new pcl::PointCloud<PointT>);
    pcl::PointCloud<SHOT1344>::Ptr prev_descriptors(
       new pcl::PointCloud<SHOT1344>);
    pcl::PointCloud<SHOT1344>::Ptr next_descriptors(
       new pcl::PointCloud<SHOT1344>);
#ifdef _OPENMP
#pragma omp parallel sections
#endif
    {
#ifdef _OPENMP
#pragma omp section
#endif
       {
          this->estimatePointCloudNormals<int>(
             prev_cloud, prev_normals, k, true);
          this->keypoints3D(prev_keypoints, prev_cloud);
          this->features3D(prev_descriptors, prev_cloud,
                           prev_normals, prev_keypoints);
       }
#ifdef _OPENMP
#pragma omp section
#endif
       {
          this->estimatePointCloudNormals<int>(
             next_cloud, next_normals, k, true);
          this->keypoints3D(next_keypoints, next_cloud);
          this->features3D(next_descriptors, next_cloud,
                           next_normals, next_keypoints);
       }
    }

    std::cout << "------------------------------------------------"  << "\n";
    std::cout << "DEBUG: "  << "\n";
    std::cout << prev_cloud->size() << ", " << prev_keypoints->size()
              << ", "
              << prev_normals->size() << ", " << prev_descriptors->size()
              << "\n";
    std::cout << next_cloud->size() << ", " << next_keypoints->size()
              << ", "
              << next_normals->size() << ", " << next_descriptors->size()
              << "\n";
    
    // finding correspondances
    pcl::CorrespondencesPtr correspondences (new pcl::Correspondences);
    pcl::KdTreeFLANN<SHOT1344>::Ptr matcher(new pcl::KdTreeFLANN<SHOT1344>);
    matcher->setInputCloud(prev_descriptors);
    std::vector<int> indices;
    std::vector<float> distances;
    const float thresh = 0.50f;
    for (int i = 0; i < next_descriptors->size(); i++) {
       bool is_finite = true;
       for (int j = 0; j < FEATURE_DIM; j++) {
          if (isnan(next_descriptors->points[i].descriptor[j])) {
             is_finite = false;
             break;
          }
       }
       if (is_finite) {
          int found = matcher->nearestKSearch(next_descriptors->at(i), 1,
                                              indices, distances);
          if (found == 1 && distances[0] < thresh) {
             pcl::Correspondence corresp(indices[0], i, distances[0]);
             correspondences->push_back(corresp);
          }
       }
    }


    std::cout << "correspondance: " << correspondences->size()  << "\n";
    
    // corresponding using Hough like voting
    pcl::PointCloud<RFType>::Ptr prev_rf (new pcl::PointCloud<RFType> ());
    pcl::PointCloud<RFType>::Ptr next_rf (new pcl::PointCloud<RFType> ());
    pcl::BOARDLocalReferenceFrameEstimation<PointT, Normal, RFType> rf_est;
    rf_est.setFindHoles(true);
    rf_est.setRadiusSearch(0.02f);
    rf_est.setInputCloud(prev_keypoints);
    rf_est.setInputNormals(prev_normals);
    rf_est.setSearchSurface(prev_cloud);
    rf_est.compute(*prev_rf);
    rf_est.setInputCloud(next_keypoints);
    rf_est.setInputNormals(next_normals);
    rf_est.setSearchSurface(next_cloud);
    rf_est.compute(*next_rf);

    pcl::Hough3DGrouping<PointT, PointT, RFType, RFType> clusterer;
    clusterer.setHoughBinSize(0.02f);
    clusterer.setHoughThreshold(4.0f);
    clusterer.setUseInterpolation(true);
    clusterer.setUseDistanceWeight(false);
    
    clusterer.setInputCloud(prev_keypoints);
    clusterer.setInputRf(prev_rf);
    clusterer.setSceneCloud(next_keypoints);
    clusterer.setSceneRf(next_rf);
    clusterer.setModelSceneCorrespondences(correspondences);

    std::vector<pcl::Correspondences> clustered_corrs;
    std::vector<Eigen::Matrix4f,
                Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    clusterer.recognize(rototranslations, clustered_corrs);

    std::cout << "\n NUM_CORRESPONDANCE: " << clustered_corrs.size()  << "\n";
    
    if (clustered_corrs.size() > 0) {
       return true;
    } else {
       return false;
    }
}

template<class T>
void ObjectRegionEstimation::estimatePointCloudNormals(
    const pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<Normal>::Ptr normals,
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

void ObjectRegionEstimation::clusterFeatures(
    std::vector<pcl::PointIndices> &all_indices,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<Normal>::Ptr descriptors,
    const int min_size, const float max_distance) {
    if (descriptors->empty()) {
      ROS_ERROR("ERROR: EMPTY FEATURES.. SKIPPING CLUSTER SRV");
      return;
    }
    interactive_segmentation::Feature3DClustering srv;
    for (int i = 0; i < descriptors->size(); i++) {
      jsk_recognition_msgs::Histogram hist;
      hist.histogram.push_back(cloud->points[i].x);
      hist.histogram.push_back(cloud->points[i].y);
      hist.histogram.push_back(cloud->points[i].z);
      hist.histogram.push_back(descriptors->points[i].normal_x);
      hist.histogram.push_back(descriptors->points[i].normal_y);
      hist.histogram.push_back(descriptors->points[i].normal_z);
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
       all_indices.resize(max_label + 1);
       for (int i = 0; i < srv.response.labels.size(); i++) {
          int index = srv.response.labels[i];
          if (index > -1) {
             all_indices[index].indices.push_back(i);
          }
       }
    } else {
       ROS_ERROR("SRV CLIENT CALL FAILED");
       return;
    }
}

/**
 * NOT USED
 */
void ObjectRegionEstimation::removeStaticKeypoints(
    pcl::PointCloud<PointI>::Ptr prev_keypoints,
    pcl::PointCloud<PointI>::Ptr curr_keypoints,
    const float threshold) {
    pcl::PointCloud<PointI>::Ptr keypoints(new pcl::PointCloud<PointI>);
    const int size = prev_keypoints->size();
    int match_index[size];
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_) shared(match_index)
#endif
    for (int i = 0; i < prev_keypoints->size(); i++) {
       match_index[i] = -1;
       for (int j = 0; j < curr_keypoints->size(); j++) {
          double distance = pcl::distances::l2(
             prev_keypoints->points[i].getVector4fMap(),
             curr_keypoints->points[j].getVector4fMap());
          if (distance < threshold) {
             match_index[i] = j;
          }
       }
    }
    
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "object_region_estimation");
    ObjectRegionEstimation ore;
    ros::spin();
    return 0;
}
