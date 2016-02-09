// Copyright (C) 2016 by Krishneel Chaudhary, JSK Lab,a
// The University of Tokyo, Japan

#include <interactive_segmentation/object_region_estimation.h>

ObjectRegionEstimation::ObjectRegionEstimation() :
    num_threads_(8), is_prev_ok(false), min_cluster_size_(100) {
    this->prev_cloud_ = pcl::PointCloud<PointT>::Ptr(
       new pcl::PointCloud<PointT>);
    // this->onInit();

    cv::Mat img1 = cv::imread("/home/krishneel/Desktop/frame0000.jpg");
    cv::Mat img2 = cv::imread("/home/krishneel/Desktop/frame0001.jpg");
    
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

    if (img1.empty() || img2.empty()) {
       std::cout << "EMPTY"  << "\n";
    }

    float angle = 0;
    float magnitude = 0;
    magnitudeAngleThresholds(angle, magnitude, img1, img2);
    this->sceneFlow(cloud, img1, img2, cloud);
    
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
    this->sync_prev_ = boost::make_shared<message_filters::Synchronizer<
      SyncPolicyPrev> >(100);
    this->sync_prev_->connectInput(sub_image_prev_, sub_cloud_prev_);
    this->sync_prev_->registerCallback(boost::bind(
        &ObjectRegionEstimation::callbackPrev, this, _1, _2));

    
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_normal_.subscribe(this->pnh_, "input_normals", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(sub_cloud_, sub_normal_);
    this->sync_->registerCallback(boost::bind(
                               &ObjectRegionEstimation::callback,
                               this, _1, _2));
}

void ObjectRegionEstimation::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_indices_.unsubscribe();
    this->sub_normal_.unsubscribe();
}

/**
 * table removed image and object region point cloud
 */
void ObjectRegionEstimation::callbackPrev(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
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
    /*
    if (cloud->empty() || cv_ptr->image.empty()) {
      is_prev_ok = false;
    } else {
      pcl::copyPointCloud<PointT, PointT>(*cloud, *prev_cloud_);
      this->prev_image_ = cv_ptr->image.clone();
      is_prev_ok = true;
    }
    */

    std::cout << "NOW RUNNING....."  << "\n";
    
    // delete this
    header_ = cloud_msg->header;
    if (cv_ptr->image.empty()) {
      std::cout << "EMPTY"  << "\n";
      return;
    }
    if (!is_prev_ok) {
      is_prev_ok = true;
      prev_image_ = cv_ptr->image.clone();
      return;
    }
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    sceneFlow(filtered, prev_image_, cv_ptr->image, cloud);
    
    // prev_image_ = cv_ptr->image.clone();

    // std::cout << "SLEEPING"  << "\n";
    // ros::Duration(10).sleep();
}

// also sub to the origial image
/**
 * current table remove and origal cloud
 */
void ObjectRegionEstimation::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::PointCloud2::ConstPtr &normal_msg) {
    pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *tmp_cloud);
    pcl::PointCloud<Normal>::Ptr tmp_normals(new pcl::PointCloud<Normal>);
    pcl::fromROSMsg(*normal_msg, *tmp_normals);
    if (tmp_cloud->size() != tmp_normals->size()) {
       ROS_ERROR("INCORRECT INPUT SIZE");
       return;
    }
    this->header_ = cloud_msg->header;
    
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<Normal>::Ptr normals(new pcl::PointCloud<Normal>);
    for (int i = 0; i < tmp_cloud->size(); i++) {
       PointT pt = tmp_cloud->points[i];
       Normal nt = tmp_normals->points[i];
       if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z) &&
           !isnan(nt.normal_x) && !isnan(nt.normal_y) && !isnan(nt.normal_z)) {
          cloud->push_back(pt);
          normals->push_back(tmp_normals->points[i]);
       }
    }

    
    // 1) scene flow
    // 2) filter and select
    // 3) 
    
    
    // pcl::PointCloud<PointI>::Ptr keypoints(new pcl::PointCloud<PointI>);
    // this->keypoints3D(keypoints, cloud);

    // pcl::PointCloud<SHOT352>::Ptr descriptors(new pcl::PointCloud<SHOT352>);
    // this->features3D(descriptors, cloud, normals, keypoints);

    std::vector<pcl::PointIndices> all_indices;
    this->clusterFeatures(all_indices, cloud, normals, 5, 0.5);

    std::cout << "Cluster Size: " << all_indices.size() << "\n";
    
    jsk_recognition_msgs::ClusterPointIndices ros_indices;
    ros_indices.cluster_indices = pcl_conversions::convertToROSPointIndices(
        all_indices, cloud_msg->header);
    ros_indices.header = cloud_msg->header;
    pub_indices_.publish(ros_indices);
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    // this->pub_cloud_.publish(ros_cloud);
}

void ObjectRegionEstimation::sceneFlow(
    pcl::PointCloud<PointT>::Ptr filtered, const cv::Mat prev_image,
    const cv::Mat next_image, const pcl::PointCloud<PointT>::Ptr cloud) {
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
          if ((val > 90.0 && val < 360.0) && magnitude.at<float>(j, i) > 5.0f) {
             val /= 360.0f;
             orientation.at<cv::Vec3b>(j, i) [0]=  val * 255.0f;
             image.at<uchar>(j, i) += next_img.at<uchar>(j, i);
             
             // filtered->push_back(cloud->points[i + (j * angle.cols)]);
             
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
    pcl::PointCloud<SHOT352>::Ptr descriptors,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<Normal>::Ptr normals,
    const pcl::PointCloud<PointI>::Ptr keypoints) {
    pcl::PointCloud<PointT>::Ptr keypoints_xyz(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointI, PointT>(*keypoints, *keypoints_xyz);
    pcl::SHOTEstimationOMP<PointT, Normal, SHOT352> shot;
    shot.setSearchSurface(cloud);
    shot.setInputCloud(keypoints_xyz);
    shot.setInputNormals(normals);
    shot.setRadiusSearch(0.02f);
    shot.compute(*descriptors);
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

void ObjectRegionEstimation::magnitudeAngleThresholds(
    float &angle_thres, float &magn_thresh,
    const cv::Mat prev_image, const cv::Mat next_image) {
    double quality_level = 0.01;
    double min_distance = 10;
    int block_size = 3;
    int max_corners = 500;
    cv::Mat prev_img = prev_image.clone();
    cv::Mat next_img = next_image.clone();
    if (prev_image.type() != 0) {
      cv::cvtColor(prev_image, prev_img, CV_BGR2GRAY);
    }
    if (next_image.type() != 0) {
      cv::cvtColor(next_image, next_img, CV_BGR2GRAY);
    }
    cv::GoodFeaturesToTrackDetector gftt(max_corners, quality_level, min_distance);
    std::vector<cv::KeyPoint> prev_keypoints;
    gftt.detect(prev_img, prev_keypoints);
    std::vector<cv::KeyPoint> next_keypoints;
    gftt.detect(next_img, next_keypoints);

    cv::SurfDescriptorExtractor extractor;
    cv::Mat prev_descriptor;
    extractor.compute(prev_img, prev_keypoints, prev_descriptor);
    cv::Mat next_descriptor;
    extractor.compute(next_img, next_keypoints, next_descriptor);

    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(prev_descriptor, next_descriptor, matches);
    
    double max_dist = 0;
    double min_dist = 100;
    for(int i = 0; i < prev_descriptor.rows; i++) {
      double dist = matches[i].distance;
      if (dist < min_dist) {
        min_dist = dist;
      }
      if (dist > max_dist) {
        max_dist = dist;
      }
    }
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < prev_descriptor.rows; i++) {
      if (matches[i].distance < 3*min_dist ) {
        good_matches.push_back(matches[i]);
      }
    }
    std::vector<float> x_coords;
    std::vector<float> y_coords;
    float angle_avg = 0.0f;
    float magn_avg = 0.0f;
    for (int i = 0; i < good_matches.size(); i++) {
      cv::Point2f pt = next_keypoints[good_matches[i].trainIdx].pt -
          prev_keypoints[good_matches[i].queryIdx].pt;
      magn_avg += (std::sqrt(pt.x * pt.x + pt.y * pt.y));
      angle_avg += (std::atan2(pt.y, pt.x) * 180.0 / M_PI);
    }
    magn_avg /= static_cast<float>(good_matches.size());
    angle_avg /= static_cast<float>(good_matches.size());
    
    std::cout << "AVG: " << magn_avg << "\t" << angle_avg  << "\n";
    
    
    cv::Mat img_matches;
    cv::drawMatches(prev_img, prev_keypoints, next_img, next_keypoints,
                    good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("match", img_matches);
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "object_region_estimation");
    ObjectRegionEstimation ore;
    ros::spin();
    return 0;
}
