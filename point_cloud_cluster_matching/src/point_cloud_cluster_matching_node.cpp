
#include <point_cloud_cluster_matching/point_cloud_cluster_matching.h>
#include <vector>

PointCloudClusterMatching::PointCloudClusterMatching() :
    processing_counter_(0),
    depth_counter(0),
    min_object_size_(100),
    max_distance_(1.5f) {
    this->subscribe();
    this->onInit();
}

void PointCloudClusterMatching::onInit() {

    this->pub_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>(
       "/manipulated_cluster/output/cloud_cluster", sizeof(char));
    
    this->pub_signal_ = pnh_.advertise<point_cloud_scene_decomposer::signal>(
       "/manipulated_cluster/output/signal", sizeof(char));
}

void PointCloudClusterMatching::subscribe() {

    this->sub_signal_ = this->pnh_.subscribe(
      "input_signal", sizeof(char),
      &PointCloudClusterMatching::signalCallback, this);
    this->sub_grip_end_pose_ = this->pnh_.subscribe(
       "input_gripper_end_pose", sizeof(char),
       &PointCloudClusterMatching::gripperEndPoseCallback, this);
    this->sub_manip_cluster_ = this->pnh_.subscribe(
       "input_manip_cluster", sizeof(char),
       &PointCloudClusterMatching::manipulatedClusterCallback, this);
    this->sub_indices_ = this->pnh_.subscribe(
       "input_indices", sizeof(char),
       &PointCloudClusterMatching::indicesCallback, this);
    this->sub_cloud_prev_ = this->pnh_.subscribe(
       "input_cloud_prev", sizeof(char),
       &PointCloudClusterMatching::cloudPrevCallback, this);
    this->sub_image_ = pnh_.subscribe(
       "input_image", sizeof(char),
       &PointCloudClusterMatching::imageCallback, this);
    this->sub_image_prev_ = pnh_.subscribe(
       "input_image_prev", sizeof(char),
       &PointCloudClusterMatching::imagePrevCallback, this);
    this->sub_mask_ = pnh_.subscribe(
       "input_mask", sizeof(char),
       &PointCloudClusterMatching::imageMaskCallback, this);
    this->sub_mask_prev_ = pnh_.subscribe(
       "input_mask_prev", sizeof(char),
       &PointCloudClusterMatching::imageMaskPrevCallback, this);
    this->sub_cloud_ = this->pnh_.subscribe(
       "input_cloud", sizeof(char),
       &PointCloudClusterMatching::cloudCallback, this);
}

void PointCloudClusterMatching::unsubscribe() {
    this->sub_cloud_.shutdown();
    this->sub_indices_.shutdown();
    this->sub_signal_.shutdown();
    this->sub_manip_cluster_.shutdown();
    this->sub_grip_end_pose_.shutdown();
}

void PointCloudClusterMatching::signalCallback(
    const point_cloud_scene_decomposer::signal &signal_msg) {
    this->signal_ = signal_msg;
}

void PointCloudClusterMatching::gripperEndPoseCallback(
    const geometry_msgs::Pose & end_pose_msg) {
    this->gripper_pose_ = end_pose_msg;
}

void PointCloudClusterMatching::manipulatedClusterCallback(
    const std_msgs::Int16 &manip_cluster_index_msg) {
    this->manipulated_cluster_index_ = manip_cluster_index_msg.data;
}


/**
 * subscribers to the current and previous RGB Images
 */
void PointCloudClusterMatching::imageCallback(
    const sensor_msgs::Image::ConstPtr &image_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(
           image_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    this->image_ = cv_ptr->image.clone();
}

void PointCloudClusterMatching::imagePrevCallback(
    const sensor_msgs::Image::ConstPtr &image_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(
           image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    this->image_prev_ = cv_ptr->image.clone();
}


/**
 * subscibers to the previous and current masked images
 */
void PointCloudClusterMatching::imageMaskCallback(
    const sensor_msgs::Image::ConstPtr &image_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(
          image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return;
    }
    cv::Mat tmp = cv_ptr->image.clone();
    cv::cvtColor(tmp, tmp, CV_BGR2GRAY);
    cv::threshold(
       tmp, this->image_mask_, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
}

void PointCloudClusterMatching::imageMaskPrevCallback(
    const sensor_msgs::Image::ConstPtr &image_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(
          image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return;
    }
    cv::Mat tmp = cv_ptr->image.clone();
    cv::cvtColor(tmp, tmp, CV_BGR2GRAY);
    cv::threshold(
       tmp, this->image_mask_prev_, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
}

void PointCloudClusterMatching::indicesCallback(
    const jsk_recognition_msgs::ClusterPointIndices &indices_msgs) {
    this->all_indices.clear();
    for (int i = 0; i < indices_msgs.cluster_indices.size(); i++) {
       pcl::PointIndices indices;
       indices.indices = indices_msgs.cluster_indices[i].indices;
       this->all_indices.push_back(indices);
    }
}

/**
 * subscribers to previous and current point clouds
 */
void PointCloudClusterMatching::cloudPrevCallback(
    const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    cloud_prev_ = pcl::PointCloud<PointT>::Ptr(
       new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud_prev_);
    // this->prev_cloud_clusters.clear();
    // this->objectCloudClusters(
    //    cloud_prev, this->all_indices, this->prev_cloud_clusters);
}

void PointCloudClusterMatching::cloudCallback(
    const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if (cloud->empty() || image_.empty() ||
        image_mask_.empty() || this->image_mask_prev_.empty()) {
       ROS_ERROR("-- EMPTY CLOUD CANNOT BE PROCESSED.");
       return;
    }
    if (this->signal_.command == 3 &&
        this->signal_.counter == this->processing_counter_) {

       const float manip_distance_threshold = 0.05f;
       const float max_manip_distance = 0.30f;
       cv::Mat matte_image = cv::Mat::zeros(image_mask_.size(), CV_8U);
       for (int j = 0; j < image_mask_.rows; j++) {
          for (int i = 0; i < image_mask_.cols; i++) {
             if (image_mask_.at<uchar>(j, i) !=
                 (image_mask_prev_.at<uchar>(j, i))) {
                matte_image.at<uchar>(j, i) = 255;
             }
             int index = i + (j * matte_image.cols);
             if (cloud->points[index].z > this->max_distance_) {
                cloud->points[index].z = 0.0f;
                cloud->points[index].y = 0.0f;
                cloud->points[index].x = 0.0f;
             }
             if (this->cloud_prev_->points[index].z > this->max_distance_) {
                this->cloud_prev_->points[index].z = 0.0f;
                this->cloud_prev_->points[index].y = 0.0f;
                this->cloud_prev_->points[index].x = 0.0f;
             }
             float x1 = this->cloud_prev_->points[index].x;
             float y1 = this->cloud_prev_->points[index].y;
             float z1 = this->cloud_prev_->points[index].z;
             float x2 = cloud->points[index].x;
             float y2 = cloud->points[index].y;
             float z2 = cloud->points[index].z;
             if (!isnan(x1) && !isnan(y1) && !isnan(z1) &&
                 !isnan(x2) && !isnan(y2) && !isnan(z2)) {
                x2 -= x1;
                y2 -= y1;
                z2 -= z1;
                float manip_dist = std::sqrt(
                   (x2 * x2) + (y2 * y2) + (z2 * z2));
                if (manip_dist > manip_distance_threshold &&
                    manip_dist < max_manip_distance) {
                   matte_image.at<uchar>(j, i) = 255;
                }
             }
          }
       }
       
       this->cvMorphologicalOperations(matte_image, matte_image, true, 1);
       this->cvMorphologicalOperations(matte_image, matte_image, false, 1);
       // this->contourSmoothing(matte_image);

       cv::Mat current_roi;
       cv::Mat cur_mask_invert;
       cv::bitwise_not(image_mask_, cur_mask_invert);
       cv::bitwise_and(matte_image, cur_mask_invert, current_roi);
    
       pcl::PointIndices::Ptr filtered_indices(new pcl::PointIndices);
       for (int j = 0; j < current_roi.rows; j++) {
          for (int i = 0; i < current_roi.cols; i++) {
             int index = i + (j * current_roi.cols);
             if (static_cast<int>(
                    current_roi.at<uchar>(j, i)) == 255 &&
                 cloud->points[index].z < this->max_distance_) {
                filtered_indices->indices.push_back(index);
             }
          }
       }

       if (!filtered_indices->indices.empty())  {
          pcl::ExtractIndices<PointT>::Ptr eifilter(
             new pcl::ExtractIndices<PointT>);
          eifilter->setInputCloud(cloud);
          eifilter->setIndices(filtered_indices);
          eifilter->filter(*cloud);

          // this->clusterMovedObjectROI(cloud, filtered_indices);
          
          sensor_msgs::PointCloud2 ros_cloud;
          pcl::toROSMsg(*cloud, ros_cloud);
          ros_cloud.header = cloud_msg->header;

          this->publishing_cloud.data.clear();
          this->publishing_cloud = ros_cloud;
       
          this->pub_cloud_.publish(ros_cloud);
       }
       this->processing_counter_++;
       
       // cv::imshow("current roi", current_roi);
       // cv::imshow("Matte", matte_image);
       // cv::imshow("mask", image_mask_);
       cv::imshow("inverted", cur_mask_invert);
       cv::imshow("prev_mask", image_mask_);
       cv::waitKey(3);
       
    } else {
       if (this->processing_counter_ != 0) {
          this->publishing_cloud.header = cloud_msg->header;
          this->pub_cloud_.publish(publishing_cloud);
       } else {
          ROS_WARN("-- NOT PROCESSING");
       }
    }
    point_cloud_scene_decomposer::signal pub_sig;
    pub_sig.header = cloud_msg->header;
    pub_sig.command = 1;
    pub_sig.counter = this->processing_counter_;
    this->pub_signal_.publish(pub_sig);

    std::cout << "Counter: " << pub_sig.command << "\t"
              << pub_sig.counter   << std::endl;
    }

void PointCloudClusterMatching::clusterMovedObjectROI(
    pcl::PointCloud<PointT>::Ptr in_cloud,
    pcl::PointIndices::Ptr indices,
    const int min_cluster_size) {
    if (in_cloud->empty() || indices->indices.empty()) {
       ROS_WARN("-- Not object Region for Clustering...");
       return;
    }
    pcl::search::KdTree<PointT>::Ptr tree(
       new pcl::search::KdTree<PointT>);
    tree->setInputCloud(in_cloud);
    pcl::EuclideanClusterExtraction<PointT> euclidean_clustering;
    euclidean_clustering.setClusterTolerance(0.02);
    euclidean_clustering.setMinClusterSize(min_cluster_size);
    euclidean_clustering.setMaxClusterSize(25000);
    euclidean_clustering.setSearchMethod(tree);
    euclidean_clustering.setInputCloud(in_cloud);
    // euclidean_clustering.setIndices(indices);
    std::vector<pcl::PointIndices> cluster_indices;
    euclidean_clustering.extract(cluster_indices);
    
    /* TODO(ADD THE CODE TO CHECK THE BEST CLUSTER NEADR LAST GRIP POSITION)*/
    int index = -1;
    int icounter = 0;
    int cluster_size = min_cluster_size;
    for (std::vector<pcl::PointIndices>::iterator it = cluster_indices.begin();
         it != cluster_indices.end(); it++) {
       if (it->indices.size() > cluster_size) {
          cluster_size = it->indices.size();
          index = icounter;
       }
       icounter++;
    }

    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    for (int i = 0; i < cluster_indices[index].indices.size(); i++) {
       int idx = cluster_indices[index].indices.at(i);
       cloud_filtered->push_back(in_cloud->points[idx]);
    }
    in_cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*cloud_filtered, *in_cloud);
    
    /*
    indices = pcl::PointIndices::Ptr(new pcl::PointIndices);
    if (index != -1) {
       indices->indices = cluster_indices[index].indices;
    }
    */
}

void PointCloudClusterMatching::contourSmoothing(
    cv::Mat &mask_img) {
    if (mask_img.empty()) {
       ROS_ERROR("--NO REGION IN EMPTY MASK");
       return;
    }
    cv::Mat canny_output;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Canny(mask_img, canny_output, 50, 150, 3);
    findContours(canny_output, contours, hierarchy,
                 CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    const double MIN_AREA = 20;
    mask_img = cv::Mat::zeros(canny_output.size(), CV_8U);
    for (int i = 0; i < contours.size(); i++) {
       double area = cv::contourArea(contours[i]);
       if (area > MIN_AREA) {
          cv::drawContours(
             mask_img, contours, i, cv::Scalar(255), CV_FILLED);
       }
    }
}


void PointCloudClusterMatching::cvMorphologicalOperations(
    const cv::Mat &img, cv::Mat &erosion_dst,
    bool is_errode, int erosion_size) {
    if (img.empty()) {
       ROS_ERROR("Cannnot perfrom Morphological Operations on empty image....");
       return;
    }
    int erosion_const = 2;
    int erosion_type = cv::MORPH_ELLIPSE;
    cv::Mat element = cv::getStructuringElement(erosion_type,
       cv::Size(erosion_const * erosion_size + sizeof(char),
                erosion_const * erosion_size + sizeof(char)),
       cv::Point(erosion_size, erosion_size));
    if (is_errode) {
       cv::erode(img, erosion_dst, element);
    } else {
       cv::dilate(img, erosion_dst, element);
    }
}

















void PointCloudClusterMatching::extractObjectROIIndices(
    cv::Rect_<int>  &rect,
    pcl::PointIndices::Ptr cluster_indices,
    const cv::Size image_size) {
    if (rect.x < 0) {
       rect.x = 0;
    }
    if (rect.y < 0) {
       rect.y = 0;
    }
    if (rect.x + rect.width > image_size.width) {
       rect.width -= ((rect.x + rect.width) - image_size.width);
    }
    if (rect.y + rect.height > image_size.height) {
       rect.height -= ((rect.y + rect.height) - image_size.height);
    }
    for (int j = rect.y; j < (rect.y + rect.height); j++) {
       for (int i = rect.x; i < (rect.x + rect.width); i++) {
          int index = i + (j * image_size.width);
          cluster_indices->indices.push_back(index);
       }
    }
}

void PointCloudClusterMatching::objectCloudClusters(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const std::vector<pcl::PointIndices> &cluster_indices,
    std::vector<pcl::PointCloud<PointT>::Ptr> &cloud_clusters) {
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    for (std::vector<pcl::PointIndices>::const_iterator it =
            cluster_indices.begin(); it != cluster_indices.end(); it++) {
       pcl::PointCloud<PointT>::Ptr c_cloud(new pcl::PointCloud<PointT>);
       pcl::PointIndices::Ptr indices(new pcl::PointIndices());
       indices->indices = it->indices;
       extract.setIndices(indices);
       extract.setNegative(false);
       extract.filter(*c_cloud);
       cloud_clusters.push_back(c_cloud);
    }
}

void PointCloudClusterMatching::createImageFromObjectClusters(
    const std::vector<pcl::PointCloud<PointT>::Ptr> &cloud_clusters,
    const sensor_msgs::CameraInfo::ConstPtr camera_info,
    const cv::Mat &prev_image,
    std::vector<cv::Mat> &image_patches,
    std::vector<cv::Rect_<int> > &regions) {
    image_patches.clear();
    const int min_size = 20;
    for (std::vector<pcl::PointCloud<PointT>::Ptr>::const_iterator it =
            cloud_clusters.begin(); it != cloud_clusters.end(); it++) {
       if (!(*it)->empty()) {
          cv::Mat mask;
          cv::Mat img_out = this->projectPointCloudToImagePlane(
             *it, camera_info, mask);
          cv::Mat interpolate_img = this->interpolateImage(img_out, mask);
          cv::Rect_<int> rect;
          this->getObjectRegionMask(interpolate_img, rect);
          if (rect.width > min_size && rect.height > min_size) {
             cv::Mat roi = prev_image(rect).clone();
             image_patches.push_back(roi);
          }
          cv::imshow("image", interpolate_img);
          cv::waitKey(3);
       }
    }
}

void PointCloudClusterMatching::getObjectRegionMask(
    cv::Mat &image, cv::Rect_<int> &rect) {
    if (image.empty()) {
       ROS_ERROR("-- CONTOUR OF EMPTY IMAGE");
       return;
    }
    if (image.type() != CV_8U) {
       cv::cvtColor(image, image, CV_BGR2GRAY);
    }
    cv::threshold(image, image, 10, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    cv::Mat cols;
    cv::Mat rows;
    int min_x = 640;
    int max_x = 0;
    int min_y = 640;
    int max_y = 0;
    const int downsize = 2;
    cv::resize(image, image, cv::Size(image.cols/downsize,
                                      image.rows/downsize));
    for (int j = 0; j < image.rows; j++) {
       for (int i = 0; i < image.cols; i++) {
          if (static_cast<int>(image.at<uchar>(j, i)) ==  0) {
             if (i < min_x) {
                min_x = i;
             }
             if (j < min_y) {
                min_y = j;
             }
             if (i > max_x) {
                max_x = i;
             }
             if (j > max_y) {
                max_y = j;
             }
          }
       }
    }
    const int padding = 8;
    min_x *= downsize;
    min_y *= downsize;
    max_y *= downsize;
    max_x *= downsize;
    rect = cv::Rect_<int>(static_cast<int>(min_x - padding),
                          static_cast<int>(min_y - padding),
                          static_cast<int>(max_x + padding - min_x),
                          static_cast<int>(max_y + padding - min_y));
    cv::resize(image, image, cv::Size(
                  image.cols * downsize, image.rows * downsize));
    cv::cvtColor(image, image, CV_GRAY2BGR);
    cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
}

void PointCloudClusterMatching::extractKeyPointsAndDescriptors(
    const cv::Mat &image, cv::Mat &descriptor,
    std::vector<cv::KeyPoint> &keypoints) {
    cv::Ptr<cv::FeatureDetector> detector =
       cv::FeatureDetector::create("SIFT");
    detector->detect(image, keypoints);
    // cv::SurfDescriptorExtractor extractor;
    cv::SiftDescriptorExtractor extractor;
    extractor.compute(image, keypoints, descriptor);
}

void PointCloudClusterMatching::computeFeatureMatch(
    const cv::Mat &img_object, const cv::Mat img_scene,
    const cv::Mat &descriptors_object, const cv::Mat &descriptors_scene,
    const std::vector<cv::KeyPoint> &keypoints_object,
    const std::vector<cv::KeyPoint> &keypoints_scene,
    cv::Rect_<int> &rect) {
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_object, descriptors_scene, matches);

    double max_dist = 0; double min_dist = 100;
    for (int i = 0; i < descriptors_object.rows; i++) {
       double dist = matches[i].distance;
       if (dist < min_dist) min_dist = dist;
       if (dist > max_dist) max_dist = dist;
    }

    std::cout << "Min Max distance: " << min_dist
              << "\t" << max_dist << std::endl;
    
    std::vector<cv::DMatch> good_matches;
    // good_matches = matches;


    const int SHIFT_DISTANCE = 10;
    // for (int i = 0; i < descriptors_object.rows; i++) {
       
    //    if (matches[i].distance < 2 * min_dist) {
    //       good_matches.push_back(matches[i]);
    //    }
    // }

    
    for (int i = 0; i < matches.size(); i++) {
       cv::Point obj_pt = keypoints_object[matches[i].queryIdx].pt;
       cv::Point sce_pt = keypoints_scene[matches[i].trainIdx].pt;
       double dist = cv::norm(cv::Mat(obj_pt), cv::Mat(sce_pt));

       if (dist > 20) {
          good_matches.push_back(matches[i]);
       }
    }

    
    if (good_matches.size() > 3) {
       cv::Mat img_matches;
       cv::drawMatches(
          img_object, keypoints_object, img_scene, keypoints_scene,
          good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
          std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
       std::vector<cv::Point2f> obj;
       std::vector<cv::Point2f> scene;
       for (int i = 0; i < good_matches.size(); i++) {
            obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
            scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
        }
       cv::Mat H = cv::findHomography(obj, scene, CV_RANSAC);
       std::vector<cv::Point2f> obj_corners(4);
       obj_corners[0] = cv::Point(0, 0);
       obj_corners[1] = cv::Point(img_object.cols, 0);
       obj_corners[2] = cv::Point(img_object.cols, img_object.rows);
       obj_corners[3] = cv::Point(0, img_object.rows);
       std::vector<cv::Point2f> scene_corners(4);
       cv::perspectiveTransform(obj_corners, scene_corners, H);

       const int PADDING = 8;
       rect = this->detectMatchROI(
          img_scene, scene_corners[0], scene_corners[1],
          scene_corners[2], scene_corners[3]);
       rect.x -= PADDING;
       rect.y -= PADDING;
       rect.width += PADDING * 2;
       rect.height += PADDING * 2;
       if (rect.x < 0) {
          rect.x = 0;
       }
       if (rect.y < 0) {
          rect.y = 0;
       }
       if (rect.x + rect.width > img_scene.cols) {
          rect.width -= ((rect.x + rect.width) - img_scene.cols);
       }
       if (rect.y + rect.height > img_scene.rows) {
          rect.height -= ((rect.y + rect.height) - img_scene.rows);
       }
       cv::Scalar color = cv::Scalar(0, 255, 0);
       cv::line(img_matches, scene_corners[0] + cv::Point2f(img_object.cols, 0),
                scene_corners[1] + cv::Point2f(img_object.cols, 0), color, 4);
       cv::line(img_matches, scene_corners[1] + cv::Point2f(img_object.cols, 0),
            scene_corners[2] + cv::Point2f(img_object.cols, 0), color, 4);
       cv::line(img_matches, scene_corners[2] + cv::Point2f(img_object.cols, 0),
            scene_corners[3] + cv::Point2f(img_object.cols, 0), color, 4);
       cv::line(img_matches, scene_corners[3] + cv::Point2f(img_object.cols, 0),
            scene_corners[0] + cv::Point2f(img_object.cols, 0), color, 4);
       cv::Mat temp = img_scene.clone();
       cv::imshow("Good Matches & Object detection", img_matches);
       cv::rectangle(temp, rect, cv::Scalar(255, 0, 255), 2);
       imshow("scene", temp);
    }
}


cv::Rect_<int> PointCloudClusterMatching::detectMatchROI(
    const cv::Mat &image, cv::Point2f &corner_top_left,
    cv::Point2f &corner_top_right, cv::Point2f &corner_bottom_right,
    cv::Point2f &corner_bottom_left) {
    
    if (corner_top_left.x < 0) {
        corner_top_left.x = 0;
    }
    if (corner_top_left.y < 0) {
        corner_top_left.y = 0;
    }
    if (corner_top_left.x < 0 && corner_bottom_left.x < 0) {
        corner_top_left.x = 0;
        corner_bottom_left. x = 0;
    }
    if (corner_top_left.y < 0 && corner_top_right.y < 0) {
        corner_top_left.y = 0;
        corner_top_right.y = 0;
    }
    if (corner_top_right.y < 0) {
        corner_top_right.y = 0;
    }
    if (corner_top_right.x > image.cols) {
        corner_top_right.x = image.cols;
    }
    if (corner_top_right.x > image.cols && corner_bottom_right.x > image.cols) {
        corner_top_right.x = image.cols;
        corner_bottom_right.x = image.cols;
    }
    if (corner_bottom_right.y > image.rows) {
        corner_bottom_right.y = image.rows;
    }
    if (corner_bottom_right.y > image.rows &&
        corner_bottom_left.y > image.rows) {
        corner_bottom_right.y = image.rows;
        corner_bottom_left.y = image.rows;
    }
    if (corner_bottom_left.x < 0) {
        corner_bottom_left.x = 0;
    }
    if (corner_bottom_left.y > image.rows) {
        corner_bottom_left.y = image.rows;
    }
    int y_coord_top = std::min(corner_top_left.y, corner_top_right.y);
    int y_coord_bot = std::min(corner_bottom_left.y, corner_bottom_right.y);
    int hei_t = std::max(corner_top_left.y, corner_top_right.y);
    int hei_b = std::max(corner_bottom_left.y, corner_bottom_right.y);
    int top_left_y = std::min(y_coord_bot, y_coord_top);
    int height_l = std::max(hei_b, hei_t);
    int height = std::abs(top_left_y - height_l);
    int x_coord_top = std::min(corner_top_left.x, corner_top_right.x);
    int x_coord_bot = std::min(corner_bottom_right.x, corner_bottom_left.x);
    int wid_t = std::max(corner_top_left.x, corner_top_right.x);
    int wid_b = std::max(corner_bottom_right.x, corner_bottom_left.x);
    int top_left_x = std::min(x_coord_bot, x_coord_top);
    int width_l = std::max(wid_b, wid_t);
    int width = std::abs(top_left_x - width_l);
    if ((width + top_left_x) > image.cols) {
        width = (width + top_left_x) - image.cols;
    }
    if ((height + top_left_y) > image.rows) {
        height = (height + top_left_y) - image.rows;
    }
    if (top_left_x < 0) {
        top_left_x = 0;
    }
    if (top_left_y < 0) {
        top_left_y = 0;
    }
    cv::Rect_<int> rect = cv::Rect_<int>(top_left_x, top_left_y, width, height);
    return rect;
}




int main(int argc, char *argv[]) {

    ros::init(argc, argv, "point_cloud_cluster_matching");
    PointCloudClusterMatching pccm;
    ros::spin();
    return 0;
}
