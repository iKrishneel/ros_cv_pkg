
#include <incremental_icp_registeration/feature_based_registeration.h>

FeatureBasedRegisteration::FeatureBasedRegisteration() {

    this->prev_cloud = pcl::PointCloud<PointT>::Ptr(
       new pcl::PointCloud<PointT>);
    this->prev_nnan_cloud = pcl::PointCloud<PointT>::Ptr(
       new pcl::PointCloud<PointT>);
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

    bool feature_3d = false;
    if (feature_3d) {
       pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(
          new pcl::PointCloud<pcl::PointNormal>);
       this->estimatePointCloudNormals<int>(
          nnan_cloud, cloud_normals, 10, true);
       pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints(
          new pcl::PointCloud<pcl::PointWithScale>);
       this->getPointCloudKeypoints(nnan_cloud, cloud_normals, keypoints);
       pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(
        new pcl::PointCloud<pcl::FPFHSignature33>());
       this->computePointFPFH(nnan_cloud, cloud_normals, keypoints, features);

       if (!this->prev_features->points.empty()) {
          bool is_pcl_outlier = false;
          if (is_pcl_outlier) {
             pcl::registration::CorrespondenceEstimation<
                pcl::FPFHSignature33, pcl::FPFHSignature33, double> estimate;
             estimate.setInputSource(this->prev_features);
             estimate.setInputTarget(features);
             boost::shared_ptr<pcl::Correspondences> correspondences(
                new pcl::Correspondences);
             estimate.determineCorrespondences(*correspondences);

             boost::shared_ptr<pcl::Correspondences> out_correspondences(
                new pcl::Correspondences);

             int rej_sac_max_dist = 0.01;
             int rej_sac_max_iter = 5000;
             pcl::registration::CorrespondenceRejectorSampleConsensus<
                PointT> corr_rej_sac;
             corr_rej_sac.setInputSource(this->prev_cloud);
             corr_rej_sac.setInputTarget(nnan_cloud);
             corr_rej_sac.setInlierThreshold(rej_sac_max_dist);
             corr_rej_sac.setMaximumIterations(rej_sac_max_iter);
             corr_rej_sac.getRemainingCorrespondences(
                *correspondences, *out_correspondences);


             Eigen::Matrix4f transform_res_sac =
                corr_rej_sac.getBestTransformation();
             
             std::cout << "Size Matching: " << out_correspondences->size()
                       << "\t" << correspondences->size() << std::endl;
          }  else {
             boost::shared_ptr<pcl::Correspondences> correspondences(
                new pcl::Correspondences);
             this->featureCorrespondenceEstimate(
                prev_features, features, correspondences);
          
          }
       }
    }
    
       
    if (!this->prev_descriptor_.empty()) {
       std::vector<cv::KeyPoint> keypoints;
       cv::Mat descriptor;
       this->keypointsFrom2DImage(cloud, this->image, keypoints, descriptor);
       boost::shared_ptr<pcl::Correspondences> correspondences(
             new pcl::Correspondences);
       this->featureCorrespondenceEstimate2D(
          cloud, prev_image, prev_keypoints_, prev_descriptor_,
          image, keypoints, descriptor, correspondences);
       
       boost::shared_ptr<pcl::Correspondences> out_correspondences(
          new pcl::Correspondences);
       float rej_sac_max_dist = 0.1;
       int rej_sac_max_iter = 5000;
       pcl::registration::CorrespondenceRejectorSampleConsensus<
          PointT> corr_rej_sac;
       corr_rej_sac.setInputSource(this->prev_nnan_cloud);
       corr_rej_sac.setInputTarget(nnan_cloud);
       corr_rej_sac.setInlierThreshold(rej_sac_max_dist);
       corr_rej_sac.setMaximumIterations(rej_sac_max_iter);
       corr_rej_sac.getRemainingCorrespondences(
          *correspondences, *out_correspondences);
       Eigen::Matrix4f transform_res_sac =
          corr_rej_sac.getBestTransformation();

       std::cout << "Transform: \n" << transform_res_sac << std::endl;
       
       std::cout << "Size Matching: " << out_correspondences->size()
                 << "\t" << correspondences->size() << std::endl;
       
       this->draw2DFinalCorrespondence(
          prev_image, prev_keypoints_, image, keypoints, out_correspondences);
       
       
    } else {
       ROS_WARN("SETTING INITIAL FEATURES");
       this->keypointsFrom2DImage(
          cloud, this->image, this->prev_keypoints_, this->prev_descriptor_);
       this->prev_image = this->image.clone();
       pcl::copyPointCloud<PointT, PointT>(*cloud, *prev_cloud);
       pcl::copyPointCloud<PointT, PointT>(*nnan_cloud, *prev_nnan_cloud);
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
    const pcl::PointCloud<PointT>::Ptr cloud, const cv::Mat &img,
    std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptor) {
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, CV_BGR2GRAY);
    cv::Ptr<cv::FeatureDetector> detector =
       cv::FeatureDetector::create("ORB");
    // cv::SiftFeatureDetector detector(1000);
    detector->detect(gray_img, keypoints);
    cv::SiftDescriptorExtractor extractor;
    extractor.compute(gray_img, keypoints, descriptor);

    
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

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
       new pcl::PointCloud<pcl::Normal>);
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


void FeatureBasedRegisteration::featureCorrespondenceEstimate2D(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const cv::Mat prev_img, const std::vector<cv::KeyPoint> prev_keypoints,
    const cv::Mat prev_descriptor, const cv::Mat img,
    const std::vector<cv::KeyPoint> keypoints, const cv::Mat descriptor,
    boost::shared_ptr<pcl::Correspondences> correspondences) {
    if (prev_descriptor.empty() || descriptor.empty()) {
       ROS_ERROR("CANNOT COMPUTE CORRESPONDENCE OF EMPTY FEATURES");
       return;
    }
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(prev_descriptor, descriptor, matches);

    double max_dist = 0;
    double min_dist = FLT_MAX;
    for (int i = 0; i < prev_descriptor.rows; i++) {
       double dist = matches[i].distance;
       if (dist < min_dist) {
          min_dist = dist;
       }
       if (dist > max_dist) {
          max_dist = dist;
       }
    }
    
    double threshold = 5;
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < prev_descriptor.rows; i++) {
       // if (matches[i].distance < std::max(
       //        threshold * min_dist, 0.01 * max_dist)) {
          pcl::Correspondence corr;
          cv::Point pt_q = prev_keypoints[matches[i].queryIdx].pt;
          cv::Point pt_t = keypoints[matches[i].trainIdx].pt;
          int index_q = pt_q.x + (pt_q.y * image.cols);
          int index_t = pt_t.x + (pt_t.y * image.cols);

          if (!isnan(prev_cloud->points[index_q].x) &&
              !isnan(prev_cloud->points[index_q].y) &&
              !isnan(prev_cloud->points[index_q].z) &&
              !isnan(cloud->points[index_t].x) &&
              !isnan(cloud->points[index_t].y) &&
              !isnan(cloud->points[index_t].z)) {
             double dist = pcl::distances::l2(
                cloud->points[index_t].getVector4fMap(),
                prev_cloud->points[index_q].getVector4fMap());
             
             corr.index_query = index_q;
             corr.index_match = index_t;
             corr.distance = dist;
             correspondences->push_back(corr);
             good_matches.push_back(matches[i]);
             //}
       }
    }
    std::cout << "PREV SIZE: "  << prev_descriptor.size() << std::endl;
    std::cout << "GOOD MATCH: " << correspondences->size() << std::endl;
}


void FeatureBasedRegisteration::draw2DFinalCorrespondence(
    const cv::Mat prev_img, std::vector<cv::KeyPoint> prev_keypoints,
    const cv::Mat img, std::vector<cv::KeyPoint> keypoints,
    const boost::shared_ptr<pcl::Correspondences> correspondences) {
    if (prev_keypoints.empty() || keypoints.empty()) {
       ROS_ERROR("EMPTY DATA");
       return;
    }
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < correspondences->size(); i++) {
       cv::DMatch m;
       m.queryIdx = correspondences->operator[](i).index_query;
       m.trainIdx = correspondences->operator[](i).index_match;
       
    }
    
    cv::Mat img_matches;
    cv::drawMatches(prev_img, prev_keypoints, img, keypoints,
                    good_matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("match", img_matches);
}


void FeatureBasedRegisteration::featureCorrespondenceEstimate(
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr source,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target,
    boost::shared_ptr<pcl::Correspondences> correspondences) {
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
    double threshold = 2;
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < src_descriptor.rows; i++) {
      if (matches[i].distance < std::max(
             threshold * min_dist, 0.1 * max_dist)) {
        pcl::Correspondence corr;
        corr.index_query = matches[i].queryIdx;
        corr.index_match = matches[i].trainIdx;
        corr.distance = matches[i].distance;
        correspondences->push_back(corr);
        good_matches.push_back(matches[i]);
      }
    }
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
      cv::normalize(
         descriptor, descriptor, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
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
