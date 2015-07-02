
#include <point_cloud_scene_decomposer/point_cloud_cluster_matching.h>
#include <vector>

PointCloudClusterMatching::PointCloudClusterMatching() :
    processing_counter_(0),
    depth_counter(0),
    min_object_size_(100),
    max_distance_(1.5f) {
   
    this->detector_ = cv::FeatureDetector::create("FAST");
    this->descriptor_ = cv::DescriptorExtractor::create("ORB");

    this->known_object_bboxes_.clear();
    this->subscribe();
    this->onInit();
}

void PointCloudClusterMatching::onInit() {

    this->pub_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>(
       "/manipulated_cluster/output/cloud_cluster", sizeof(char));
    this->pub_indices_ = pnh_.advertise<
       jsk_recognition_msgs::ClusterPointIndices>(
          "/manipulated_cluster/output/indices", sizeof(char));
    this->pub_known_bbox_ = pnh_.advertise<
       jsk_recognition_msgs::BoundingBoxArray>(
          "/manipulated_cluster/output/known_bounding_boxes", sizeof(char));
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
     this->sub_bbox_ = pnh_.subscribe(
        "input_bbox", 1,
        &PointCloudClusterMatching::boundingBoxCallback, this);
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
    const geometry_msgs::PoseStamped & end_pose_msg) {
    this->gripper_pose_ = end_pose_msg;
}

/**
 * subscriber to the bounding_box_filter of current frame
 */
void PointCloudClusterMatching::boundingBoxCallback(
    const jsk_recognition_msgs::BoundingBoxArray &bba_msg) {
    this->bbox_ = bba_msg;
}

/**
 * subscriber to which cluster was manipulated
 */
void PointCloudClusterMatching::manipulatedClusterCallback(
    const std_msgs::Int64 &manip_cluster_index_msg) {
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
           image_msg, sensor_msgs::image_encodings::BGR8);
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
    this->all_indices_.clear();
    for (int i = 0; i < indices_msgs.cluster_indices.size(); i++) {
       pcl::PointIndices indices;
       indices.indices = indices_msgs.cluster_indices[i].indices;
       this->all_indices_.push_back(indices);
    }
}

/**
 * subscribers to previous and current point clouds
 */
void PointCloudClusterMatching::cloudPrevCallback(
    const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    this->cloud_prev_ = pcl::PointCloud<PointT>::Ptr(
       new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud_prev_);
    // this->prev_cloud_clusters.clear();
    // this->objectCloudClusters(
    //    cloud_prev, this->all_indices_, this->prev_cloud_clusters);
}

void PointCloudClusterMatching::cloudCallback(
    const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    /*
    std::cout << "Incoming Signal: " << this->signal_.command << "\t"
              << this->signal_.counter << std::endl;
    */
    if (cloud->empty() /*|| image_.empty()*/ ||
        image_mask_.empty() || this->image_mask_prev_.empty()) {
       std::cout << "Size: " << cloud->size() << "\t" << image_mask_.size()
                 << "\t" << image_mask_prev_.size() << std::endl;
       ROS_ERROR("-- EMPTY CLOUD CANNOT BE PROCESSED.");
       return;
    }
    
    
    if (this->signal_.command == 3 &&
        this->signal_.counter == this->processing_counter_
/*&&
  /*this->manipulated_cluster_index_ != -1*/) {

       cv::Mat prev_masked = image_prev_.clone();
       cv::Mat cur_masked = image_.clone();
       for (int j = 0; j < image_mask_.rows; j++) {
          for (int i = 0; i < image_mask_.cols; i++) {
             if (image_mask_.at<uchar>(j, i) != 0) {
                cur_masked.at<cv::Vec3b>(j, i)[0] = 0;
                cur_masked.at<cv::Vec3b>(j, i)[1] = 0;
                cur_masked.at<cv::Vec3b>(j, i)[2] = 0;
             }
             if (image_mask_prev_.at<uchar>(j, i) != 0) {
                prev_masked.at<cv::Vec3b>(j, i)[0] = 0;
                prev_masked.at<cv::Vec3b>(j, i)[1] = 0;
                prev_masked.at<cv::Vec3b>(j, i)[2] = 0;
             }
          }

       }
       // FeatureInfo info;
       // forwardBackwardMatchingAndFeatureCorrespondance(
       //    image_prev_, image_, info);

       
       pcl::PointCloud<PointT>::Ptr in_cloud(new pcl::PointCloud<PointT>);
       pcl::copyPointCloud<PointT, PointT>(*cloud, *in_cloud);
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
       cloud->clear();
       pcl::copyPointCloud<PointT, PointT>(*in_cloud, *cloud);
       
       this->cvMorphologicalOperations(matte_image, matte_image, true, 2);
       this->cvMorphologicalOperations(matte_image, matte_image, false, 2);
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
                 !isnan(cloud->points[index].x) &&
                 !isnan(cloud->points[index].y) &&
                 !isnan(cloud->points[index].z)
                 /*cloud->points[index].z < this->max_distance_*/) {
                filtered_indices->indices.push_back(index);
             }
          }
       }

       cv::imwrite("/home/krishneel/Desktop/roi.jpg", current_roi);
       
       
       if (!filtered_indices->indices.empty())  {
          // publish the indices
          std::vector<pcl::PointIndices> obj_index(1);
          // this->clusterMovedObjectROI(cloud, filtered_indices, obj_index);
          obj_index[0].indices = filtered_indices->indices;

          
          pcl::ExtractIndices<PointT>::Ptr eifilter(
             new pcl::ExtractIndices<PointT>);
          eifilter->setInputCloud(cloud);
          // eifilter->setIndices(filtered_indices);
          filtered_indices->indices.clear();
          filtered_indices->indices = obj_index[0].indices;
          eifilter->setIndices(filtered_indices);
          eifilter->filter(*cloud);
          
          sensor_msgs::PointCloud2 ros_cloud;
          pcl::toROSMsg(*cloud, ros_cloud);
          ros_cloud.header = cloud_msg->header;
          this->pub_cloud_.publish(ros_cloud);

          this->publishing_cloud.data.clear();
          this->publishing_cloud = ros_cloud;
          
          
          pcl::io::savePCDFileASCII (
             "/home/krishneel/Desktop/test_pcd.pcd", *cloud);

          
          if (!obj_index.empty()) {
             // current manipulated object centroid
             Eigen::Vector4f centroid;
             pcl::compute3DCentroid<PointT, float>(
                *cloud, obj_index[0], centroid);
             float ct_x = static_cast<float>(centroid[0]);
             float ct_y = static_cast<float>(centroid[1]);
             float ct_z = static_cast<float>(centroid[2]);
             if (!isnan(ct_x) && !isnan(ct_y) && !isnan(ct_z)) {
                this->known_object_bboxes_.push_back(
                   Eigen::Vector3f(ct_x, ct_y, ct_z));
             }      

             std::cout << "\n\n KNOW SIZE: " <<
                known_object_bboxes_.size() <<"\n\n" << std::endl;
                
             jsk_recognition_msgs::ClusterPointIndices ros_indices;
             ros_indices.cluster_indices = this->convertToROSPointIndices(
                obj_index, cloud_msg->header);
             ros_indices.header = cloud_msg->header;
             this->pub_indices_.publish(ros_indices);
             
             this->publishing_indices.cluster_indices.insert(
                this->publishing_indices.cluster_indices.end(),
                ros_indices.cluster_indices.begin(),
                ros_indices.cluster_indices.end());
          
          }

          std::cout << "COMPUTING BOX..."  << std::endl;
          jsk_recognition_msgs::BoundingBoxArray known_object_bbox_array;
          this->getKnownObjectRegion(
             this->known_object_bboxes_, known_object_bbox_array, 0.2f);

          
          known_object_bbox_array.header = cloud_msg->header;
          this->pub_known_bbox_.publish(known_object_bbox_array);
          this->publishing_known_bbox = known_object_bbox_array;
          
          // std::cout << "Cluster Size: " << obj_index.size() << "\t"
          //           << publishing_known_bbox.boxes.size() << std::endl;
       }
       this->processing_counter_++;
       /*
       cv::imshow("current roi", current_roi);
       cv::imshow("Matte", matte_image);
       cv::imshow("mask", image_mask_);
       cv::imshow("inverted", cur_mask_invert);
       cv::imshow("prev_mask", image_mask_);
       cv::waitKey(3);
       */
    } else {
       if (this->processing_counter_ != 0  /* &&
           /*this->manipulated_cluster_index_ != -1*/) {
          // ROS_WARN("-- CLUSTERING NODE PUBLISHING OLD TOPICS");
          this->publishing_cloud.header = cloud_msg->header;
          this->publishing_indices.header = cloud_msg->header;
          this->publishing_known_bbox.header = cloud_msg->header;
          
          this->pub_known_bbox_.publish(this->publishing_known_bbox);
          this->pub_cloud_.publish(this->publishing_cloud);
          this->pub_indices_.publish(this->publishing_indices);
       }
 /* else if (this->manipulated_cluster_index_ == -1) {
          ROS_INFO("\nALL OBJECTS ON THE TABLE IS LABELED\n");
          }*/
    }
    point_cloud_scene_decomposer::signal pub_sig;
    pub_sig.header = cloud_msg->header;
    pub_sig.command = 1;
    pub_sig.counter = this->processing_counter_;
    this->pub_signal_.publish(pub_sig);

    /*
    std::cout << " --- Counter: " << pub_sig.command << "\t"
              << pub_sig.counter  << "\t internal Counter: " <<
              this->processing_counter_ << std::endl;
    */
}

void PointCloudClusterMatching::clusterMovedObjectROI(
    pcl::PointCloud<PointT>::Ptr in_cloud,
    pcl::PointIndices::Ptr indices,
    std::vector<pcl::PointIndices> &moved_indices,
    const int min_cluster_size,
    const float min_distance_threshold) {
    if (in_cloud->empty() || indices->indices.empty()) {
       ROS_WARN("-- Not object Region for Clustering...");
       return;
    }
    pcl::search::KdTree<PointT>::Ptr tree(
       new pcl::search::KdTree<PointT>);
    tree->setInputCloud(in_cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> euclidean_clustering;
    euclidean_clustering.setClusterTolerance(0.02);
    euclidean_clustering.setMinClusterSize(min_cluster_size);
    euclidean_clustering.setMaxClusterSize(25000);
    euclidean_clustering.setSearchMethod(tree);
    euclidean_clustering.setInputCloud(in_cloud);
    // euclidean_clustering.setIndices(indices);
    euclidean_clustering.extract(cluster_indices);

    std::cout << "\n ----- # of Cluster: " << cluster_indices.size()
              << "\n"<< std::endl;
    
    int index = -1;
    int icounter = 0;
    float end_effector_dist = FLT_MAX;
    for (std::vector<pcl::PointIndices>::iterator it = cluster_indices.begin();
         it != cluster_indices.end(); it++) {
       if (it->indices.size() > min_cluster_size) {
          Eigen::Vector4f centroid;
          pcl::compute3DCentroid<PointT, float>(
             *in_cloud, *it, centroid);
          float ct_x = static_cast<float>(centroid[0]);
          float ct_y = static_cast<float>(centroid[1]);
          float ct_z = static_cast<float>(centroid[2]);
          if (!isnan(ct_x) && !isnan(ct_y) && !isnan(ct_z)) {
             float dist = std::sqrt(
                std::pow((ct_x - this->gripper_pose_.pose.position.x), 2) +
                std::pow((ct_y - this->gripper_pose_.pose.position.y), 2) +
                std::pow((ct_z - this->gripper_pose_.pose.position.z), 2));
             if (dist < end_effector_dist && dist < min_distance_threshold) {
                end_effector_dist = dist;
                index = icounter;
             }
          }
       }
       icounter++;
    }
    if (index != -1) {
       pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
       for (int i = 0; i < cluster_indices[index].indices.size(); i++) {
          int idx = cluster_indices[index].indices.at(i);
          cloud_filtered->push_back(in_cloud->points[idx]);
       }
       moved_indices.clear();
       moved_indices.push_back(cluster_indices[index]);
       in_cloud->clear();
       pcl::copyPointCloud<PointT, PointT>(*cloud_filtered, *in_cloud);
    }
}


/**
 * determines all the known object region
 TODO(Add the feature matching)
 */
void PointCloudClusterMatching::getKnownObjectRegion(
    const std::vector<Eigen::Vector3f> &manipulated_obj_centroids,
    jsk_recognition_msgs::BoundingBoxArray &known_object_bbox,
    const float min_assignment_threshold) {
    for (int j = 0; j < manipulated_obj_centroids.size(); j++) {
       float distance = FLT_MAX;
       int object_index = -1;
       float b_x = manipulated_obj_centroids[j][0];
       float b_y = manipulated_obj_centroids[j][1];
       float b_z = manipulated_obj_centroids[j][2];

       std::cout << "Centroid: " << b_x << "\t"
                 << b_y << "\t" << b_z << std::endl;
       
       for (int i = 0; i < this->bbox_.boxes.size(); i++) {
          float c_x = this->bbox_.boxes[i].pose.position.x +
             this->bbox_.boxes[i].dimensions.x / 2;
          float c_y = this->bbox_.boxes[i].pose.position.y +
             this->bbox_.boxes[i].dimensions.y / 2;
          float c_z = this->bbox_.boxes[i].pose.position.z +
             this->bbox_.boxes[i].dimensions.z / 2;
          float dist = std::sqrt(
             std::pow((c_x - b_x), 2) +
             std::pow((c_y - b_y), 2) +
             std::pow((c_z - b_z), 2));

          std::cout << "Bbox: " << c_x << "\t"
                 << c_y << "\t" << c_z << std::endl;
          
           std::cout << "Distance: " << dist << "\t"
                     << object_index << "\n" << std::endl;
          
          if (dist < distance &&
              dist < min_assignment_threshold) {
             distance = dist;
             object_index = i;
          }
       }
       if (object_index != -1) {
          known_object_bbox.boxes.push_back(this->bbox_.boxes[object_index]);
       }
    }
    // std::cout << "Know Size: " << known_object_bbox.boxes.size() << std::endl;
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

std::vector<pcl_msgs::PointIndices>
PointCloudClusterMatching::convertToROSPointIndices(
    const std::vector<pcl::PointIndices> cluster_indices,
    const std_msgs::Header& header) {
    std::vector<pcl_msgs::PointIndices> ret;
    for (size_t i = 0; i < cluster_indices.size(); i++) {
       pcl_msgs::PointIndices ros_msg;
       ros_msg.header = header;
       ros_msg.indices = cluster_indices[i].indices;
       ret.push_back(ros_msg);
    }
    return ret;
}


void PointCloudClusterMatching::buildImagePyramid(
    const cv::Mat &frame,
    std::vector<cv::Mat> &pyramid) {
    cv::Mat gray = frame.clone();
    cv::Size winSize = cv::Size(5, 5);
    int maxLevel = 5;
    bool withDerivative = true;
    cv::buildOpticalFlowPyramid(
       gray, pyramid, winSize, maxLevel, withDerivative,
       cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, true);
    
}

void PointCloudClusterMatching::getOpticalFlow(
    const cv::Mat &frame, const cv::Mat &prevFrame,
    std::vector<cv::Point2f> &nextPts, std::vector<cv::Point2f> &prevPts,
    std::vector<uchar> &status) {
    cv::Mat gray, grayPrev;
    cv::cvtColor(prevFrame, grayPrev, CV_BGR2GRAY);
    cv::cvtColor(frame, gray, CV_BGR2GRAY);
    std::vector<cv::Mat> curPyramid;
    std::vector<cv::Mat> prevPyramid;
    buildImagePyramid(frame, curPyramid);
    buildImagePyramid(prevFrame, prevPyramid);
    std::vector<float> err;
    nextPts.clear();
    status.clear();
    nextPts.resize(prevPts.size());
    status.resize(prevPts.size());
    err.resize(prevPts.size());
    cv::Size winSize = cv::Size(5, 5);
    int maxLevel = 3;
    cv::TermCriteria criteria = cv::TermCriteria(
       cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    int flags = 0;
    cv::calcOpticalFlowPyrLK(
       prevPyramid, curPyramid, prevPts, nextPts, status,
       err, winSize, maxLevel, criteria, flags);
    cv::Mat iFrame = prevFrame.clone();
}


void PointCloudClusterMatching::forwardBackwardMatchingAndFeatureCorrespondance(
    const cv::Mat img1, const cv::Mat img2, FeatureInfo &info) {
    std::vector<cv::Point2f> nextPts;
    std::vector<cv::Point2f> prevPts;
    std::vector<cv::Point2f> backPts;
    // cv::GaussianBlur(img1, img1, cv::Size(5, 5), 1);
    // cv::GaussianBlur(img2, img2, cv::Size(5, 5), 1);
    cv::Mat gray;
    cv::Mat grayPrev;
    cv::cvtColor(img1, grayPrev, CV_BGR2GRAY);
    cv::cvtColor(img2, gray, CV_BGR2GRAY);
    std::vector<cv::KeyPoint> keypoints_prev;
    this->detector_->detect(grayPrev, keypoints_prev);
    for (int i = 0; i < keypoints_prev.size(); i++) {
       prevPts.push_back(keypoints_prev[i].pt);
    }
    std::vector<uchar> status;
    std::vector<uchar> status_back;
    this->getOpticalFlow(img2, img1, nextPts, prevPts, status);
    this->getOpticalFlow(img1, img2, backPts, nextPts, status_back);
    std::vector<float> fb_err;
    for (int i = 0; i < prevPts.size(); i++) {
       cv::Point2f v = backPts[i] - prevPts[i];
       fb_err.push_back(sqrt(v.dot(v)));
    }
    float THESHOLD = 10;
    for (int i = 0; i < status.size(); i++) {
       status[i] = (fb_err[i] <= THESHOLD) & status[i];
    }
    std::vector<cv::KeyPoint> keypoints_next;
    for (int i = 0; i < prevPts.size(); i++) {
       cv::Point2f ppt = prevPts[i];
       cv::Point2f npt = nextPts[i];
       double distance = cv::norm(cv::Mat(ppt), cv::Mat(npt));
       if (status[i] && distance > 5) {
          cv::KeyPoint kp;
          kp.pt = nextPts[i];
          kp.size = keypoints_prev[i].size;
          keypoints_next.push_back(kp);
       }
    }
    std::vector<cv::KeyPoint>keypoints_cur;
    this->detector_->detect(img2, keypoints_cur);
    std::vector<cv::KeyPoint> keypoints_around_region;
    for (int i = 0; i < keypoints_cur.size(); i++) {
       cv::Point2f cur_pt = keypoints_cur[i].pt;
       for (int j = 0; j < keypoints_next.size(); j++) {
          cv::Point2f est_pt = keypoints_next[j].pt;
          double distance = cv::norm(cv::Mat(cur_pt), cv::Mat(est_pt));
          if (distance < 10) {
             keypoints_around_region.push_back(keypoints_cur[i]);
          }
       }
    }
    cv::Mat descriptor_cur;
    this->descriptor_->compute(img2, keypoints_around_region, descriptor_cur);
    cv::Mat descriptor_prev;
    this->descriptor_->compute(img1, keypoints_prev, descriptor_prev);
    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher =
       cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<std::vector<cv::DMatch> > matchesAll;
    descriptorMatcher->knnMatch(descriptor_cur, descriptor_prev, matchesAll, 2);
    std::vector<cv::DMatch> match1;
    std::vector<cv::DMatch> match2;
    for (int i=0; i < matchesAll.size(); i++) {
       match1.push_back(matchesAll[i][0]);
       match2.push_back(matchesAll[i][1]);
    }
    std::vector<cv::DMatch> good_matches;
    /*
    for (int i = 0; i < matchesAll.size(); i++) {
       if (match1[i].distance < 0.7 * match2[i].distance) {
          good_matches.push_back(match1[i]);
       }
    }
    */
    // filter out approx. point match
    std::vector<cv::DMatch> final_matches;
    for (int i = 0; i < good_matches.size(); i++) {
       cv::Point2f query_pt = keypoints_around_region[
          good_matches[i].queryIdx].pt;
       cv::Point2f train_pt = keypoints_prev[
          good_matches[i].trainIdx].pt;
       double distance = cv::norm(cv::Mat(query_pt), cv::Mat(train_pt));
       if (distance > 5) {  // if points are not in smae locate
          final_matches.push_back(good_matches[i]);
       }
    }

    cv::Mat img_matches1;
    drawMatches(img2, keypoints_around_region, img1,
                keypoints_prev, final_matches, img_matches1);
    
    // cv::imshow("matches1", img_matches1);
    cv::imwrite("/home/krishneel/Desktop/match.jpg", img_matches1);
    // cv::waitKey(3);
}



int main(int argc, char *argv[]) {

    ros::init(argc, argv, "point_cloud_cluster_matching");
    PointCloudClusterMatching pccm;
    ros::spin();
    return 0;
}
