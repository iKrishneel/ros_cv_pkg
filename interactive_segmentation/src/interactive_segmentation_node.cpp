
#include <interactive_segmentation/interactive_segmentation.h>
#include <vector>

InteractiveSegmentation::InteractiveSegmentation() {
    this->subscribe();
    this->onInit();
}

void InteractiveSegmentation::onInit() {
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/interactive_segmentation/output/cloud", 1);
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/interactive_segmentation/output/image", 1);
}

void InteractiveSegmentation::subscribe() {
       this->sub_image_.subscribe(this->pnh_, "input_image", 1);
       this->sub_edge_.subscribe(this->pnh_, "input_edge", 1);
       this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
       this->sync_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy> >(100);
       sync_->connectInput(sub_image_, sub_edge_, sub_cloud_);
       sync_->registerCallback(boost::bind(&InteractiveSegmentation::callback,
                                           this, _1, _2, _3));
}

void InteractiveSegmentation::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_image_.unsubscribe();
}


void InteractiveSegmentation::callback(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const sensor_msgs::Image::ConstPtr &edge_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    boost::mutex::scoped_lock lock(this->mutex_);
    cv::Mat image = cv_bridge::toCvShare(
       image_msg, image_msg->encoding)->image;
    cv::Mat edge_img = cv_bridge::toCvShare(
       edge_msg, edge_msg->encoding)->image;
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    this->pointCloudEdge(image, edge_img, 10);
    
    
    cv_bridge::CvImage pub_img(
       image_msg->header, sensor_msgs::image_encodings::BGR8, image);
    this->pub_image_.publish(pub_img.toImageMsg());

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

void InteractiveSegmentation::pointCloudEdge(
    const cv::Mat &image, const cv::Mat &edge_img, const int contour_thresh) {
    if (image.empty()) {
       ROS_ERROR("-- Cannot eompute edge of empty image");
       return;
    }
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat cont_img = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::findContours(edge_img, contours, hierarchy, CV_RETR_LIST,
                     CV_CHAIN_APPROX_TC89_KCOS, cv::Point(0, 0));
    std::vector<std::vector<cv::Point> > selected_contours;
    for (int i = 0; i < contours.size(); i++) {
       if (cv::contourArea(contours[i]) > contour_thresh) {
          selected_contours.push_back(contours[i]);
          // drawContours(
          //    cont_img, contours, i, cv::Scalar(0, 255, 0), 1, 8,
          //    hierarchy, 0, cv::Point());
          for (int j = 0; j < contours[i].size(); j++) {
             cv::circle(cont_img, contours[i][j], 1,
                        cv::Scalar(255, 0, 0), -1);
          }
       }
    }
    std::vector<std::vector<cv::Point> > tangents;
    std::vector<std::vector<float> > orientation;
    std::vector<EdgeParam> hcurvature_pts;
    this->computeEdgeCurvature(
       cont_img, contours, tangents, orientation, hcurvature_pts);

    std::cout << contours.size() << "\t" << orientation.size() << std::endl;
    
    this->plotEdgeNormal(cont_img, contours, tangents, orientation);
    
    cv::imshow("Contours", cont_img);
    cv::waitKey(3);
}

void InteractiveSegmentation::cvMorphologicalOperations(
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

void InteractiveSegmentation::computeEdgeCurvature(
    const cv::Mat &image,
    const std::vector<std::vector<cv::Point> > &contours,
    std::vector<std::vector<cv::Point> > &tangents,
    std::vector<std::vector<float> > &orientation,
    std::vector<EdgeParam> &hcurvature_pts) {
    if (contours.empty()) {
       ROS_ERROR("-- no contours found");
       return;
    }
    cv::Mat img = image.clone();
    for (int j = 0; j < contours.size(); j++) {
       
       std::vector<cv::Point> tangent;
       cv::Point2f edge_pt = contours[j].front();
       cv::Point2f edge_tngt = contours[j].back() - contours[j][1];
       // if (edge_tngt.x >= 0 && edge_tngt.x <= image.cols &&
       // edge_tngt.y >= 0 && edge_tngt.y <= image.rows) {
          tangent.push_back(edge_tngt);
          // }
       const int neighbor_pts = 0;
       if (contours[j].size() > 2) {
          for (int i = sizeof(char) + neighbor_pts;
               i < contours[j].size() - sizeof(char) - neighbor_pts;
               i++) {
             edge_pt = contours[j][i];
             edge_tngt = contours[j][i-1-neighbor_pts] -
                contours[j][i+1+neighbor_pts];
            tangent.push_back(edge_tngt);
         }
      }
       tangents.push_back(tangent);
    }
    this->computeEdgeCurvatureOrientation(contours, tangents, orientation);
    for (int j = 0; j < tangents.size(); j++) {
       if (tangents[j].size() < 2) {
          continue;
       }
       for (int i = sizeof(char); i < tangents[j].size() - sizeof(char); i++) {
          float y1 = tangents[j][i+1].y;
          float x1 = tangents[j][i+1].x;
          float y0 = tangents[j][i-1].y;
          float x0 = tangents[j][i-1].x;
          float tang1 = 0.0f;
          float tang0 = 0.0f;
          if (x1 != 0) {
             tang1 = y1/x1;
          }
          if (x0 != 0) {
             tang0 = y0/x0;
          }
          if ((tang1 >= 0.0f  && tang0 < 0.0f) ||
              (tang1 < 0.0f && tang0 >= 0.0f)) {
             if ((abs(tang0 - tang1) < 1.0f) || (abs(tang1 - tang0) < 1.0f)) {
                continue;
             }
             EdgeParam eP;
             eP.index = contours[j][i];
             eP.orientation = orientation[j][i];
             eP.contour_position = j;
             hcurvature_pts.push_back(eP);
             // circle(img, contours[j][i], 1, cv::Scalar(255, 0, 0), 1);
             // circle(img, contours[j][i], 2, cv::Scalar(0, 255, 0), 2);
         }
      }
    }
    cv::imshow("tangent", img);
}

void InteractiveSegmentation::computeEdgeCurvatureOrientation(
    const std::vector<std::vector<cv::Point> > &contours,
    const std::vector<std::vector<cv::Point> > &contours_tangent,
    std::vector<std::vector<float> > &orientation,
    bool is_normalize) {
    for (int j = 0; j < contours_tangent.size(); j++) {
       std::vector<float> ang;
       for (int i = 0; i < contours_tangent[j].size(); i++) {
          float angle_ = 0.0f;
          if (contours_tangent[j][i].x == 0) {
             angle_ = 90.0f;
          } else {
             angle_ = atan2(contours_tangent[j][i].y,
                            contours_tangent[j][i].x) * (180/CV_PI);
          }
          if (is_normalize) {
             angle_ /= static_cast<float>(360.0);
          }
          ang.push_back(static_cast<float>(angle_));
       }
       orientation.push_back(ang);
       ang.clear();
    }
}

void InteractiveSegmentation::plotEdgeNormal(
    cv::Mat &image,
    const std::vector<std::vector<cv::Point> > &contours,
    const std::vector<std::vector<cv::Point> > &tangents,
    const std::vector<std::vector<float> > &orientation) {
    if (contours.empty() || tangents.empty() || orientation.empty()) {
       ROS_ERROR("Empty data.. skipping plot");
       return;
    }
    for (int i = 0; i < orientation.size(); i++) {
       for (int j = 0; j < orientation[i].size(); j++) {
          cv::Point pt = contours[i][j];
          cv::circle(image, pt, 1, cv::Scalar(255, 0, 255), -1);
          
          float max_norm = std::max(
            std::abs(tangents[i][j].x), std::abs(tangents[i][j].y));
         float tang_x = 0.0f;
         float tang_y = 0.0f;
         if (max_norm != 0) {
            tang_x = tangents[i][j].x / static_cast<float>(max_norm);
            tang_y = tangents[i][j].y / static_cast<float>(max_norm);
         }
         float dist = 15.0f;
         float r = sqrt((tang_y * tang_y) + (tang_x * tang_x)) * dist;
         float angle = orientation[i][j] * 360.0f * (180.0f/CV_PI);
         cv::line(image, pt, cv::Point(pt.x + r * std::cos(angle),
                                       pt.y + r * std::sin(angle)),
                  cv::Scalar(0, 255, 0), 1);
      }
    }
    cv::imshow("edge direction", image);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "interactive_segmentation");
    InteractiveSegmentation is;
    ros::spin();
    return 0;
}
