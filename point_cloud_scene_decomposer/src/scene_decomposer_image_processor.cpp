// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University of Tokyo

#include <point_cloud_scene_decomposer/scene_decomposer_image_processor.h>


SceneDecomposerImageProcessor::SceneDecomposerImageProcessor() :
    isPub(true), cell_size(cv::Size(32/2, 32/2)), it_(nh_) {

    this->rng = cv::RNG(12345);
    for (int i = 0; i < 10000; i++) {
        color[i] = cv::Scalar(
            rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }
    
    this->connectedComponents = new ConnectedComponents(30);
    
    // this->subscribe();
    // this->onInit();
}

void SceneDecomposerImageProcessor::onInit() {
    this->image_pub_ = it_.advertise("scene_decomposer/output/image", 1);
    this->depth_pub_ = it_.advertise("scene_decomposer/output/depth", 1);
}

void SceneDecomposerImageProcessor::subscribe() {
    this->image_sub_ = it_.subscribe("/camera/rgb/image_rect_color", 1,
        &SceneDecomposerImageProcessor::imageCallback, this);
    this->depth_sub_ = it_.subscribe("/camera/depth/image_rect_raw", 1,
        &SceneDecomposerImageProcessor::depthCallback, this);
}

void SceneDecomposerImageProcessor::unsubscribe() {
    this->image_sub_.shutdown();
    this->depth_sub_.shutdown();
}

SceneDecomposerImageProcessor::~SceneDecomposerImageProcessor() {
    free(this->connectedComponents);
}

/**
 * visulation
 */
void SceneDecomposerImageProcessor:: cvVisualization(
    std::vector<cvPatch<int> > &patch_label,
    const cv::Size size,
    const std::string wname) {
    cv::Mat regionMD = cv::Mat::zeros(size.height, size.width, CV_8UC3);
    int labCounter = 0;
    for (int y = 0; y < patch_label.size(); y++) {
       cv::Mat labelMD = patch_label[y].patch;
       int prev_y = patch_label[y].rect.y;
       int prev_x = patch_label[y].rect.x;
       int width = patch_label[y].rect.width;
       int height = patch_label[y].rect.height;
       // int k = patch_label[y].k;
       // if (k < 3) {
       //     cv::rectangle(
       //         regionMD, patch_label[y].rect, cv::Scalar(255, 255, 255), CV_FILLED);
       // } else {
           for (int j = 0; j < height; j++) {
               for (int i = 0; i < width; i++) {
                   int lab = static_cast<int>(labelMD.at<float>(j, i));
                   lab += labCounter;  // for unique labeling
                   regionMD.at<cv::Vec3b>(prev_y + j, prev_x + i)[2]
                       = this->color[lab].val[0];
                   regionMD.at<cv::Vec3b>(prev_y + j, prev_x + i)[1]
                       = this->color[lab].val[1];
                   regionMD.at<cv::Vec3b>(prev_y + j, prev_x + i)[0]
                       = this->color[lab].val[2];
               }
           }
           // }
       labCounter += patch_label[y].k;
    }
    /// cv::imshow(wname, regionMD);
}


/**
 * sensor image callback
 */
void SceneDecomposerImageProcessor::imageCallback(
    const sensor_msgs::ImageConstPtr &msg) {
    try {
       this->cvImgPtr = cv_bridge::toCvCopy(
          msg, sensor_msgs::image_encodings::BGR8);
       if (!this->cvImgPtr->image.empty()) {
          this->image = this->cvImgPtr->image.clone();
       }
    } catch (cv_bridge::Exception &e) {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return;
    }
}

/**
 * sensor depth callback
 */
void SceneDecomposerImageProcessor::depthCallback(
    const sensor_msgs::ImageConstPtr &msg) {
    try {
       this->cvDepPtr = cv_bridge::toCvCopy(
          msg, sensor_msgs::image_encodings::TYPE_8UC1);
       if (!this->cvDepPtr->image.empty()) {
          this->depth = this->cvDepPtr->image.clone();
       }
    } catch (cv_bridge::Exception &e) {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return;
    }
}

/**
 * processing and publish ros image
 */
void SceneDecomposerImageProcessor::publishROSImage(
    cv::Mat &img, cv::Mat &dep) {
    cv_bridge::CvImagePtr out_msg(new cv_bridge::CvImage);
    out_msg->header = this->cvImgPtr->header;
    out_msg->encoding = sensor_msgs::image_encodings::BGR8;
    out_msg->image = img;
    this->image_pub_.publish(out_msg->toImageMsg());

    out_msg = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
    out_msg->header = this->cvDepPtr->header;
    out_msg->encoding = sensor_msgs::image_encodings::TYPE_8UC1;
    out_msg->image = dep;
    this->depth_pub_.publish(out_msg->toImageMsg());

    /// cv::imshow("depth", depth);
    // cv::waitKey(3);
}


/**
 * function to extract image edge
 */
void SceneDecomposerImageProcessor::getDepthEdge(
    const cv::Mat &img,
    cv::Mat &edgeMap,
    bool isbinary) {
    if (img.empty()) {
       ROS_ERROR("Cannot find edge of empty image....");
       return;
    }
    edgeMap = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::Mat binary_edge = cv::Mat::zeros(img.size(), CV_32F);
    for (int j = 0; j < img.rows; j++) {
       for (int i = 0; i < img.cols; i++) {
          float center = static_cast<float>(img.at<uchar>(j, i) / 255.0f);
          if (center != 0) {
             float neigbor_l = static_cast<float>(
                img.at<uchar>(j, i - 1) / 255.0f);
             float neigbor_r = static_cast<float>(
                img.at<uchar>(j, i + 1) / 255.0f);
             edgeMap.at<uchar>(j, i) = static_cast<float>(
                abs((neigbor_l - neigbor_r))) * 255;
             if (isbinary) {
                if (abs((neigbor_l - neigbor_r)) > 0.025f) {
                   binary_edge.at<float>(j, i) = 1.0f;
                }
             }
          }
       }
    }
    // edgeMap = cv::Mat::zeros(img.size(), CV_32SC1);
    // edgeMap = binary_edge.clone();
    /// cv::imshow("Binary Edge", binary_edge);
    /// cv::imshow("Depth Edge", edgeMap);
    
    cv::Mat marker = cv::Mat(binary_edge.size(), CV_8UC1);
    binary_edge.convertTo(marker, CV_8UC1, 255.0f);
    edgeMap = marker.clone();
    // imshow("marker", marker);
    // cv::imwrite("/home/krishneel/Desktop/edge.jpg", marker);
}


/**
 * compute 2D image Morphological Operation (dilate/errode) image
 */
void SceneDecomposerImageProcessor::cvMorphologicalOperations(
    const cv::Mat &img,
    cv::Mat &erosion_dst,
    bool iserrode) {
    if (img.empty()) {
       ROS_ERROR("Cannnot perfrom Morphological Operations on empty image....");
       return;
    }
    int erosion_size = 3;
    int erosion_const = 2;
    int erosion_type = cv::MORPH_ELLIPSE;
    cv::Mat element = cv::getStructuringElement(
       erosion_type,
       cv::Size(erosion_const * erosion_size + sizeof(char),
                erosion_const * erosion_size + sizeof(char)),
       cv::Point(erosion_size, erosion_size));
    cv::dilate(img, erosion_dst, element);
    /// cv::imshow("Morphological Op", erosion_dst);
}

/**
 * function to extract rgb image edges
 */
void SceneDecomposerImageProcessor::getRGBEdge(
    const cv::Mat &img,
    cv::Mat &edgeMap,
    std::string type) {
    if (img.empty()) {
       ROS_ERROR("Cannot find edge of empty RGB image ...");
       return;
    }
    cv::Mat img_;
    cv::cvtColor(img, img_, CV_BGR2GRAY);
    if (type == "cvDOG") {
       cv::Mat dog1;
       cv::Mat dog2;
       cv::GaussianBlur(img_, dog1, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
       cv::GaussianBlur(img_, dog2, cv::Size(21, 21), 0, 0, cv::BORDER_DEFAULT);
       edgeMap = dog1 - dog2;
    } else if (type == "cvSOBEL") {
       int scale = 1;
       int delta = 0;
       int ddepth = CV_8UC1;
       cv::Mat grad_x;
       cv::Mat grad_y;
       cv::Mat abs_grad_x;
       cv::Mat abs_grad_y;
       cv::Sobel(img_, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
       cv::convertScaleAbs(grad_x, abs_grad_x);
       cv::Sobel(img_, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
       cv::convertScaleAbs(grad_y, abs_grad_y);
       cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edgeMap);
    } else if (type == "cvCanny") {
       cv:Canny(img_, edgeMap, 20, 100, 3);
    }
    /// cv::imshow("RGB-Edge", edgeMap);
}


/**
 * divide the edge map into spatial contigious grip
 */
void SceneDecomposerImageProcessor::cvGetLabelImagePatch(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const cv::Mat &src,
    const cv::Mat &img,
    std::vector<cvPatch<int> > &patch_label) {
    if (img.empty() || cloud->empty()) {
       ROS_ERROR("ERROR: Cannot divide empty Mat into Grid ...");
       return;
    }
    this->total_cluster = 0;
    bool isUniqueLabel = false;
    int uniqueLabel = 0;
    patch_label.clear();
    cv::Mat img_ = img.clone();
    cv::Mat pLabel = cv::Mat::zeros(img.size(), CV_32F);  // label
    std::vector<int> k_cluster;
    cvtColor(img_, img_, CV_GRAY2BGR);
    for (int j = 0; j < img.rows; j += cell_size.height) {
       for (int i = 0; i < img.cols; i += cell_size.width) {
          cv::Rect_<int> rect = cv::Rect_<int>(
             i, j, cell_size.width, cell_size.height);
          if (rect.x + rect.width <= img.cols &&
              rect.y + rect.height <= img.rows) {
             cv::Mat roi = img(rect);
             cv::Mat labelMD;
             int cluster_count = this->cvLabelImagePatch(roi, labelMD);

             // --- if work, change this routine to work on vector
             for (int jj = 0; jj < labelMD.rows; jj++) {
                for (int ii = 0; ii < labelMD.cols; ii++) {
                   pLabel.at<float>(j + jj, i + ii) =
                      labelMD.at<float>(jj, ii) + uniqueLabel;
                }
             }
             // ---
             
             k_cluster.push_back(cluster_count);
             this->total_cluster += cluster_count;
             
             if (isUniqueLabel) {
                uniqueLabel = total_cluster;
             }

             
             if (cluster_count == sizeof(char)) {
                // TODO(divide and reprocess)
             }
             /*
               cvPatch<int> ptch;
               ptch.patch = labelMD.clone();
               ptch.rect = rect;
               ptch.k = cluster_count;
               patch_label.push_back(ptch);
             */
          }
       }
    }
    
    // TODO(Remove these lines and add to above for loop)
    int icounter = 0;
    this->edgeBoundaryAssignment(
       cloud, img, pLabel, cv::Rect_<int>(0, 0, 0, 0));
    
    for (int j = 0; j < img.rows; j += cell_size.height) {
       for (int i = 0; i < img.cols; i += cell_size.width) {
          cv::Rect_<int> rect = cv::Rect_<int>(
             i, j, cell_size.width, cell_size.height);
          if (rect.x + rect.width <= img.cols &&
              rect.y + rect.height <= img.rows) {
             cv::Mat roi = pLabel(rect).clone();

             std::vector<cv::Point2i> region;
             bool is_region = false;
             cv::Mat src_roi = src(rect).clone();
             for (int y = 0; y < src_roi.rows; y++) {
                for (int x = 0; x < src_roi.cols; x++) {
                   cv::Vec3b src_roi_pix = src_roi.at<cv::Vec3b>(y, x);
                   if (src_roi_pix[0] == 255 &&
                       src_roi_pix[1] == 255 &&
                       src_roi_pix[2] == 255) {
                      // is_region = false;
                   } else {
                      is_region = true;
                      region.push_back(cv::Point2i(x, y));
                   }
                }
             }
             
             cvPatch<int> ptch;
             ptch.patch = roi.clone();
             ptch.rect = rect;
             ptch.k = k_cluster[icounter++];
             ptch.is_region = is_region;
             ptch.region = region;
             patch_label.push_back(ptch);
          }
       }
    }
}

/**
 * function to label the binary image patch using connected component analysis
 */
int SceneDecomposerImageProcessor::cvLabelImagePatch(
    const cv::Mat &in_img,
    cv::Mat &labelMD) {
    if (in_img.empty()) {
       ROS_ERROR("ERROR: No image Patch for region labeling");
       return false;
    }
    int width = in_img.cols;
    int height = in_img.rows;
    unsigned char *_img = new unsigned char[height*width];
    for (int j = 0; j < height; j++) {
       for (int i = 0; i < width; i++) {
          _img[i + (j * width)] = in_img.at<uchar>(j, i);
       }
    }
    const unsigned char *img = (const unsigned char *)_img;
    unsigned char *out_uc = new unsigned char[width*height];
    int cluster_count = this->connectedComponents->connected<
       unsigned char, unsigned char, std::equal_to<unsigned char>, bool>(
       img, out_uc, width, height, std::equal_to<unsigned char> (), false);
    // TODO(remove change to cv::Mat)
    labelMD = cv::Mat(in_img.size(), CV_32F);
    for (int j = 0; j < height; j++) {
       for (int i = 0; i < width; i++) {
          labelMD.at<float>(j, i) = static_cast<float>(
             out_uc[i + (j * width)]);
       }
    }
    free(_img);
    free(out_uc);
    return cluster_count;
}

/**
 * return the total region in a edge image
 */
int SceneDecomposerImageProcessor::getTotalClusterSize() {
    return this->total_cluster;
}


/** SUB- ORIGINIAL ORGANISED POINT CLOUD
 * function to assign edges to the pixel in foreground in local
 * neigbourhood. The pixels are argumented by the point cloud 
 */
void SceneDecomposerImageProcessor::edgeBoundaryAssignment(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const cv::Mat &edgeMap,
    cv::Mat &labelMD,
    const cv::Rect_<int> rect) {
    if (cloud->empty() || edgeMap.empty()) {
       ROS_ERROR("ERROR: No image Patch for region labeling");
       return;
    }
    const int offset_ = 3;  // distance offset from the center
    for (int j = 0; j < edgeMap.rows; j++) {
       for (int i = 0; i < edgeMap.cols; i++) {
          if (static_cast<float>(edgeMap.at<uchar>(j, i)) != 0) {
/*             // -------------------
             float min_depth = FLT_MAX;
             cv::Point2i min_idx = cv::Point2i(-1, -1);
             for (int y = j - offset_; y <= (3*offset_+j); y += offset_) {
                for (int x = i - offset_; x <= (3*offset_+i); x += offset_) {
                   if (y >= 0 && x >= 0 &&
                       y < edgeMap.rows &&
                       x < edgeMap.cols && y != j && x != i) {
                      int depth_idx = x + (y * cloud->width);
                      float depth_z = static_cast<float>(
                         cloud->points[depth_idx].z);
                      if (depth_z < min_depth) {
                         min_depth = depth_z;
                         min_idx = cv::Point2i(x, y);
                      }
                   }
                }
             }
             labelMD.at<float>(j, i) = labelMD.at<float>(min_idx.y, min_idx.x);
             // ----------------
*/
             int north = i + ((j - offset_) * edgeMap.cols);
             int south = i + ((j + offset_) * edgeMap.cols);
             int east = (i + offset_) + (j * edgeMap.cols);
             int west = (i - offset_) + (j * edgeMap.cols);
             
             float dist_n = 0.0f;
             float dist_s = 0.0f;
             float dist_e = 0.0f;
             float dist_w = 0.0f;
             float min_dist = 0.0f;
             if (j - offset_ < 0) {
                dist_s = static_cast<float>(cloud->points[south].z);
                dist_e = static_cast<float>(cloud->points[east].z);
                dist_w = static_cast<float>(cloud->points[west].z);
                min_dist = std::min(dist_s, std::min(dist_e, dist_w));
                dist_n = FLT_MAX;
             } else if (j + offset_ > edgeMap.rows) {
                dist_n = static_cast<float>(cloud->points[north].z);
                dist_e = static_cast<float>(cloud->points[east].z);
                dist_w = static_cast<float>(cloud->points[west].z);
                min_dist = std::min(dist_n, std::min(dist_e, dist_w));
                dist_s = FLT_MAX;
             }
             if (i - offset_ < 0) {
                dist_n = static_cast<float>(cloud->points[north].z);
                dist_s = static_cast<float>(cloud->points[south].z);
                dist_e = static_cast<float>(cloud->points[east].z);
                min_dist = std::min(dist_n, std::min(dist_e, dist_s));
                dist_w = FLT_MAX;
             } else if (i + offset_ > edgeMap.cols) {
                dist_n = static_cast<float>(cloud->points[north].z);
                dist_s = static_cast<float>(cloud->points[south].z);
                dist_w = static_cast<float>(cloud->points[west].z);
                min_dist = std::min(dist_n, std::min(dist_s, dist_w));
                dist_e = FLT_MAX;
             }
             if ((j - offset_ < 0) && (i - offset_ < 0)) {
                dist_s = static_cast<float>(cloud->points[south].z);
                dist_e = static_cast<float>(cloud->points[east].z);
                min_dist = std::min(dist_s, dist_e);
                dist_n = dist_w = FLT_MAX;
             }
             if ((j + offset_ > edgeMap.rows) && (i + offset_ > edgeMap.cols)) {
                dist_n = static_cast<float>(cloud->points[north].z);
                dist_w = static_cast<float>(cloud->points[west].z);
                min_dist = std::min(dist_n, dist_w);
                dist_s = dist_e = FLT_MAX;
             }
             if (min_dist == dist_n) {
                 labelMD.at<float>(j, i) = labelMD.at<float>(j - offset_, i);
              } else if (min_dist == dist_s) {
                 labelMD.at<float>(j, i) = labelMD.at<float>(j + offset_, i);
              } else if (min_dist == dist_e) {
                 labelMD.at<float>(j, i) = labelMD.at<float>(j, i + offset_);
              } else if (min_dist == dist_w) {
                 labelMD.at<float>(j, i) = labelMD.at<float>(j, i - offset_);
              }
              
          }
       }
    }
}

/**
 * process the binary edge map for region labeling
 */
void SceneDecomposerImageProcessor::cvLabelEgdeMap(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const cv::Mat &src,
    cv::Mat edgeMap,
    std::vector<cvPatch<int> > &patch_label) {
    if (edgeMap.empty() || cloud->empty()) {
       ROS_ERROR("ERROR: No image Patch for region labeling");
       return;
    }
    cv::Mat mop_img;
    this->cvMorphologicalOperations(edgeMap, mop_img);
    cvEdgeThinning(mop_img);
    // not morphological operation
    // this->cvGetLabelImagePatch(cloud, src, edgeMap, patch_label);
    this->cvGetLabelImagePatch(cloud, src, mop_img, patch_label);
    
    this->cvVisualization(patch_label, edgeMap.size());
    // cv::imshow("Thin", mop_img);
    // std::cout << "Total Size: " << this->total_cluster << std::endl;
}

/**
 * if not use morphological operation than use contour smoothing
 */
