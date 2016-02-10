// Copyright (C) 2016 by Krishneel Chaudhary, JSK Lab,a
// The University of Tokyo, Japan

#include <interactive_segmentation/object_region_estimation.h>

ObjectRegionEstimation::ObjectRegionEstimation() :
    num_threads_(8), is_prev_ok(false), min_cluster_size_(100) {
    this->go_signal_ = false;
    
    this->prev_cloud_ = pcl::PointCloud<PointT>::Ptr(
       new pcl::PointCloud<PointT>);
    this->onInit();
    /*
    cv::Mat img1 = cv::imread("/home/krishneel/Desktop/frame0000.jpg");
    cv::Mat img2 = cv::imread("/home/krishneel/Desktop/frame0001.jpg");
    
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

    if (img1.empty() || img2.empty()) {
       std::cout << "EMPTY"  << "\n";
    }
    
    this->sceneFlow(cloud, img1, img2, cloud, plane_norm_, plane_norm_);
    */
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


/**
 * table removed image and object region point cloud
 */
void ObjectRegionEstimation::callbackPrev(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::PointCloud2::ConstPtr &plane_msg) {
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
       is_prev_ok = false;
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
       is_prev_ok = true;
    }
    ROS_INFO("SEGMENTED REGION SET...");
}

void ObjectRegionEstimation::callback(
    const sensor_msgs::Image::ConstPtr &image_msg,   // table -
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,  // table -
    const sensor_msgs::PointCloud2::ConstPtr &orig_msg,   // full scene
    const geometry_msgs::PoseStamped::ConstPtr &pose_msg) {  // end-eff
    if (!is_prev_ok) {
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


    
    std::vector<pcl::PointIndices> all_indices;
    // this->clusterFeatures(all_indices, cloud, normals, 5, 0.5);

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
    for (int j = 0; j < angle.rows; j++) {
       for (int i = 0; i < angle.cols; i++) {
          float val = angle.at<float>(j, i);
          if ((val > 0.0 && val < 360.0) && magnitude.at<float>(j, i) > 5.0f) {
             val /= 360.0f;
             orientation.at<cv::Vec3b>(j, i) [0]=  val * 255.0f;
             image.at<uchar>(j, i) += next_img.at<uchar>(j, i);

             /*
             PointT pt = cloud->points[i + (j * angle.cols)];
             if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z)) {
               if (plane_norm.dot(pt.getVector3fMap() - plane_pt) >= 0.0f) {
                 filtered->push_back(pt);
               }
               }
             */
          } else {
             orientation.at<cv::Vec3b>(j, i) [1]=  val * 255.0f;
          }
       }
    }
    cv::imshow("orientation", orientation);
    cv::imshow("view", image);


    cv::Mat cflowmap;
    cv::cvtColor(next_img, cflowmap, CV_GRAY2BGR);
    drawOptFlowMap(flow, cflowmap, 5, 50, cv::Scalar(0, 255, 0));
    cv::imshow("map_flow", cflowmap);
    

    /*
    int threshold = 10;
    pnh_.getParam("thresh", threshold);

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*filtered, ros_cloud);
    ros_cloud.header = header_;
    this->pub_cloud_.publish(ros_cloud);
    */
    // cv::imshow("image", result);
    cv::waitKey(0);
}


void ObjectRegionEstimation::drawOptFlowMap(
    const cv::Mat& flow, cv::Mat& cflowmap, int step,
    double scale, const cv::Scalar& color) {
    for(int y = 0; y < cflowmap.rows; y += step) {
      for(int x = 0; x < cflowmap.cols; x += step) {
        const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
        cv::line(cflowmap, cv::Point(x,y),
                 cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
        cv::circle(cflowmap, cv::Point(cvRound(x+fxy.x),
                                       cvRound(y+fxy.y)), 1, color, -1);
      }
    }
}

void ObjectRegionEstimation::noiseClusterFilter(
    pcl::PointCloud<PointT>::Ptr object_cloud,
    pcl::PointIndices::Ptr indices,
    const pcl::PointCloud<PointT>::Ptr cloud) {
    if (cloud->empty()) {
      return;
    }
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    cluster_indices.clear();
    pcl::EuclideanClusterExtraction<PointT> euclidean_clustering;
    euclidean_clustering.setClusterTolerance(0.01f);
    euclidean_clustering.setMinClusterSize(this->min_cluster_size_);
    euclidean_clustering.setMaxClusterSize(25000);
    euclidean_clustering.setSearchMethod(tree);
    euclidean_clustering.setInputCloud(cloud);
    euclidean_clustering.extract(cluster_indices);

    // for now just select the largest cluster size
    int index = -1;
    int size = 0;
    for (int i = 0; i < cluster_indices.size(); i++) {
      if (cluster_indices[i].indices.size() > size) {
        size = cluster_indices[i].indices.size();
        index = i;
      }
    }
    pcl::ExtractIndices<PointT>::Ptr eifilter(new pcl::ExtractIndices<PointT>);
    eifilter->setInputCloud(cloud);
    *indices = cluster_indices[index];
    eifilter->setIndices(indices);
    eifilter->filter(*object_cloud);
    
    std::cout << "MAX SIZE: " << size   << "\n";
}

void ObjectRegionEstimation::keypoints3D(
    pcl::PointCloud<PointI>::Ptr keypoints,
    const pcl::PointCloud<PointT>::Ptr cloud) {
    if (cloud->empty()) {
       return;
    }
    pcl::HarrisKeypoint3D<PointT, PointI>::Ptr detector(
       new pcl::HarrisKeypoint3D<PointT, PointI>);
    detector->setNonMaxSupression(true);
    detector->setInputCloud(cloud);
    detector->setThreshold(1e-6);
    detector->compute(*keypoints);
}

void ObjectRegionEstimation::features3D(
    pcl::PointCloud<SHOT1344>::Ptr descriptors,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<Normal>::Ptr normals,
    const pcl::PointCloud<PointI>::Ptr keypoints) {
    pcl::PointCloud<PointT>::Ptr keypoints_xyz(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointI, PointT>(*keypoints, *keypoints_xyz);
    // pcl::SHOTEstimationOMP<PointT, Normal, SHOT352> shot;
    pcl::SHOTColorEstimationOMP<PointT, Normal, SHOT1344> shot;
    shot.setSearchSurface(cloud);
    shot.setInputCloud(keypoints_xyz);
    shot.setInputNormals(normals);
    shot.setRadiusSearch(0.02f);
    shot.compute(*descriptors);
}

void ObjectRegionEstimation::getHypothesis(
    const pcl::PointCloud<PointT>::Ptr prev_cloud,
    const pcl::PointCloud<PointT>::Ptr next_cloud) {
    if (prev_cloud->empty() || next_cloud->empty()) {
      ROS_ERROR("ERROR: EMPTY CANNOT CREATE OBJECT HYPOTHESIS");
      return;
    }
    const int k = 50;
    pcl::PointCloud<Normal>::Ptr prev_normals(new pcl::PointCloud<Normal>);
    pcl::PointCloud<Normal>::Ptr next_normals(new pcl::PointCloud<Normal>);
    pcl::PointCloud<PointI>::Ptr prev_keypoints(new pcl::PointCloud<PointI>);
    pcl::PointCloud<PointI>::Ptr next_keypoints(new pcl::PointCloud<PointI>);
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
    // finding correspondances
    pcl::CorrespondencesPtr correspondences (new pcl::Correspondences);
    pcl::KdTreeFLANN<SHOT1344>::Ptr matcher(new pcl::KdTreeFLANN<SHOT1344>);
    matcher->setInputCloud(prev_descriptors);
    std::vector<int> indices;
    std::vector<float> distances;
    const float thresh = 0.25f;
    for (int i = 0; i < next_descriptors->size(); i++) {
       SHOT1344 des = next_descriptors->points[i];
       int found = matcher->nearestKSearch(next_descriptors->at(i), 1,
                                           indices, distances);
       if (found == 1 && distances[0] < thresh) {
          pcl::Correspondence corresp(indices[0], i, distances[0]);
          correspondences->push_back(corresp);
       }
    }
    // corresponding using Hough like voting
    pcl::PointCloud<RFType>::Ptr prev_rf (new pcl::PointCloud<RFType> ());
    pcl::PointCloud<RFType>::Ptr next_rf (new pcl::PointCloud<RFType> ());
    pcl::BOARDLocalReferenceFrameEstimation<PointT, Normal, RFType> rf_est;
    rf_est.setFindHoles(true);
    rf_est.setRadiusSearch(0.02f);
    pcl::PointCloud<PointT>::Ptr prev_kpt_xyz(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointI, PointT>(*prev_keypoints, *prev_kpt_xyz);
    rf_est.setInputCloud(prev_kpt_xyz);
    rf_est.setInputNormals(prev_normals);
    rf_est.setSearchSurface(prev_cloud);
    rf_est.compute(*prev_rf);
    pcl::PointCloud<PointT>::Ptr next_kpt_xyz(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointI, PointT>(*next_keypoints, *next_kpt_xyz);
    rf_est.setInputCloud(next_kpt_xyz);
    rf_est.setInputNormals(next_normals);
    rf_est.setSearchSurface(next_cloud);
    rf_est.compute(*next_rf);

    pcl::Hough3DGrouping<PointT, PointT, RFType, RFType> clusterer;
    clusterer.setHoughBinSize(0.01f);
    clusterer.setHoughThreshold(5.0f);
    clusterer.setUseInterpolation(true);
    clusterer.setUseDistanceWeight(false);
    
    clusterer.setInputCloud(prev_kpt_xyz);
    clusterer.setInputRf(prev_rf);
    clusterer.setSceneCloud(next_kpt_xyz);
    clusterer.setSceneRf(next_rf);
    clusterer.setModelSceneCorrespondences(correspondences);

    std::vector<pcl::Correspondences> clustered_corrs;
    std::vector<Eigen::Matrix4f,
                Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    clusterer.recognize(rototranslations, clustered_corrs);
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
