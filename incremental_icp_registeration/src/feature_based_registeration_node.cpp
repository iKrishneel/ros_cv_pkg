
#include <incremental_icp_registeration/feature_based_registeration.h>

FeatureBasedRegisteration::FeatureBasedRegisteration() {

    this->reg_cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    this->prev_features = pcl::PointCloud<pcl::FPFHSignature33>::Ptr(
        new pcl::PointCloud<pcl::FPFHSignature33>());
    this->onInit();
}

void FeatureBasedRegisteration::onInit() {
    this->subscribe();
    this->pub_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>(
       "/feature_based_registeration/output/cloud", sizeof(char));
    this->pub_regis_ = nh_.advertise<sensor_msgs::PointCloud2>(
       "/feature_based_registeration/output/registered", sizeof(char));
}

void FeatureBasedRegisteration::subscribe() {
    this->sub_image_ = nh_.subscribe("image", 1,
       &FeatureBasedRegisteration::imageCallback, this);
    this->sub_cloud_ = nh_.subscribe("input", 1,
       &FeatureBasedRegisteration::callback, this);
}

void FeatureBasedRegisteration::unsubscribe() {
    this->sub_cloud_.shutdown();
}

void FeatureBasedRegisteration::imageCallback(
    const sensor_msgs::Image::ConstPtr &img_msg) {
    try {
       this->image = cv_bridge::toCvShare(img_msg, "bgr8")->image;
    } catch (cv_bridge::Exception &e) {
       ROS_ERROR("Could not convert from '%s' to 'bgr8'.",
                 img_msg->encoding.c_str());
    }
}

void FeatureBasedRegisteration::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if (image.empty()) {
       ROS_ERROR("INPUT IMAGE EMPTY");
       return;
    }
    
    pcl::PointCloud<PointT>::Ptr nnan_cloud(new pcl::PointCloud<PointT>);
    std::vector<int> index;
    pcl::removeNaNFromPointCloud<PointT>(*cloud, *nnan_cloud, index);

    bool is_downsample = false;
    if (is_downsample) {
      const float leaf_size = 0.01f;
      this->voxelGridFilter(nnan_cloud, nnan_cloud, leaf_size);
    }
    
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(
        new pcl::PointCloud<pcl::PointNormal>);
    this->estimatePointCloudNormals<int>(nnan_cloud, cloud_normals, 10, true);
    
    pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints(
        new pcl::PointCloud<pcl::PointWithScale>);
    bool feature_3d = false;
    if (feature_3d) {
      this->getPointCloudKeypoints(nnan_cloud, cloud_normals, keypoints);
    } else {
      this->keypointsFrom2DImage(cloud, this->image, keypoints);
    }    
    std::cout << keypoints->points.size()<< "\n";
    
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(
        new pcl::PointCloud<pcl::FPFHSignature33>());
    this->computePointFPFH(nnan_cloud, cloud_normals, keypoints, features);

    if (!this->prev_features->points.empty()) {
      // pcl::registration::CorrespondenceEstimation<
      //   pcl::FPFHSignature33, pcl::FPFHSignature33, double> estimate;
      // estimate.setInputSource(this->prev_features);
      // estimate.setInputTarget(features);
      // pcl::Correspondences correspondences;
      // estimate.determineCorrespondences(correspondences, 1.0f);
      
      this->featureCorrespondenceEstimate(prev_features, features);

    } else {pcl::CorrespondenceRejection
      ROS_WARN("SETTING INITIAL FEATURES");
      *prev_features = *features;
    }

    
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);

    sensor_msgs::PointCloud2 ros_regis;
    pcl::toROSMsg(*nnan_cloud, ros_regis);
    ros_regis.header = cloud_msg->header;
    this->pub_regis_.publish(ros_regis);
}

void FeatureBasedRegisteration::keypointsFrom2DImage(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const cv::Mat &img,
    pcl::PointCloud<pcl::PointWithScale>::Ptr extr_keypts) {
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, CV_BGR2GRAY);
    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("HARRIS");
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(gray_img, keypoints);
    for (std::vector<cv::KeyPoint>::iterator it = keypoints.begin();
         it != keypoints.end(); it++) {
      int index = static_cast<int>(it->pt.x) + (
          static_cast<int>(it->pt.y) * img.cols);
       if (!isnan(cloud->points[index].x) ||
           !isnan(cloud->points[index].y) ||
           !isnan(cloud->points[index].z)) {
         pcl::PointWithScale pws;
         pws.x = cloud->points[index].x;
         pws.y = cloud->points[index].y;
         pws.z = cloud->points[index].z;
         extr_keypts->push_back(pws);
       }
    }   
    cv::Mat draw = img.clone();
    cv::drawKeypoints(img, keypoints, draw, cv::Scalar(0, 255, 0));
    cv::imshow("image", draw);
    cv::waitKey(3);
}

void FeatureBasedRegisteration::getPointCloudKeypoints(
    const pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals,
    pcl::PointCloud<pcl::PointWithScale>::Ptr result,
    const float min_scale, const int n_octaves,
    const int n_scales_per_octave, const float min_contrast) {
    if (cloud->empty()) {
      ROS_ERROR("EMPTY POINT CLOUD");
      return;
    }
    for (int i = 0; i < cloud_normals->size(); i++) {
       cloud_normals->points[i].x = cloud->points[i].x;
       cloud_normals->points[i].y = cloud->points[i].y;
       cloud_normals->points[i].z = cloud->points[i].z;
    }
    pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree(
       new pcl::search::KdTree<pcl::PointNormal> ());
    sift.setSearchMethod(tree);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(cloud_normals);
    sift.compute(*result);
}


template<class T>
void FeatureBasedRegisteration::estimatePointCloudNormals(
    const pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<pcl::PointNormal>::Ptr normals,
    const T k, bool use_knn) const {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: The Input cloud is Empty.....");
       return;
    }
    pcl::NormalEstimationOMP<PointT, pcl::PointNormal> ne;
    ne.setInputCloud(cloud);
     ne.setNumberOfThreads(8);
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
    ne.setSearchMethod(tree);
    if (use_knn) {
        ne.setKSearch(k);
    } else {
        ne.setRadiusSearch(k);
    }    ne.compute(*normals);
}

void FeatureBasedRegisteration::computePointFPFH(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::PointNormal>::Ptr normals,
    const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature_fpfh) const {
    if (cloud->empty() || normals->empty()) {
       ROS_ERROR("-- ERROR: cannot compute FPFH");
       return;
    }

    pcl::PointCloud<PointT>::Ptr keypoints_clouds(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*keypoints, *keypoints_clouds);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    for (int i = 0; i < normals->size(); i++) {
      pcl::Normal npt;
      npt.normal_x = normals->points[i].normal_x;
      npt.normal_y = normals->points[i].normal_y;
      npt.normal_z = normals->points[i].normal_z;
      cloud_normals->push_back(npt);
    }    
    pcl::FPFHEstimationOMP<PointT, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(keypoints_clouds);
    fpfh.setSearchSurface(cloud);
    fpfh.setInputNormals(normals);
    fpfh.setNumberOfThreads(8);
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    fpfh.setSearchMethod(tree);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(
       new pcl::PointCloud<pcl::FPFHSignature33> ());
    fpfh.setRadiusSearch(0.01f);
    fpfh.compute(*fpfhs);
    *feature_fpfh = *fpfhs;
}

void FeatureBasedRegisteration::featureCorrespondenceEstimate(
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr source,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target) {
    if (source->points.empty() || target->points.empty()) {
      ROS_ERROR("CANNOT COMPUTE CORRESPONDENCE OF EMPTY FEATURES");
      return;
    }
    cv::Mat src_descriptor;
    cv::Mat tgt_descriptor;
#pragma omp parallel sections
    {
#pragma omp section
      {
        this->convertFPFHEstimationToMat(source, src_descriptor);
      }
#pragma omp section
      {
        this->convertFPFHEstimationToMat(target, tgt_descriptor);
      }
    }
    
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(src_descriptor, tgt_descriptor, matches);

    double max_dist = 0;
    double min_dist = FLT_MAX;
    for (int i = 0; i < src_descriptor.rows; i++) {
      double dist = matches[i].distance;
      if (dist < min_dist) {
        min_dist = dist;
      }
      if (dist > max_dist) {
        max_dist = dist;
      }
    }

    std::cout << min_dist << "\t" << max_dist  << "\n";
    
    std::vector<cv::DMatch> good_matches;
    double threshold = 2;
    for (int i = 0; i < src_descriptor.rows; i++) {
      if (matches[i].distance < std::max(threshold * min_dist, 0.1 * max_dist)) {
        good_matches.push_back(matches[i]);
      }
    }
    std::cout << "Good Match: " << good_matches.size() << "\n";
}


void FeatureBasedRegisteration::convertFPFHEstimationToMat(
    const pcl::PointCloud<pcl::FPFHSignature33>::Ptr source,
    cv::Mat &descriptor) {
    descriptor = cv::Mat(static_cast<int>(source->points.size()), 33, CV_32F);
    for (int i = 0; i < source->points.size(); i++) {
      for (int j = 0; j < 33; j++) {
        descriptor.at<float>(i, j) = source->points[i].histogram[j];
      }
    }
    bool is_norm = true;
    if (is_norm) {
      cv::normalize(descriptor, descriptor, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    }
}

void FeatureBasedRegisteration::voxelGridFilter(
    const pcl::PointCloud<PointT>::Ptr input,
    pcl::PointCloud<PointT>::Ptr output, const float leaf_size) {
    pcl::VoxelGrid<PointT> grid;
    grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    grid.setInputCloud(input);
    grid.filter(*output);
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "feature_based_registeration");
    FeatureBasedRegisteration fbr;
    ros::spin();
}
