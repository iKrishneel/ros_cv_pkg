
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
    
    this->pointCloudEdge(cloud, image, edge_img, 10);
    
    
    cv_bridge::CvImage pub_img(
       image_msg->header, sensor_msgs::image_encodings::BGR8, image);
    this->pub_image_.publish(pub_img.toImageMsg());

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

void InteractiveSegmentation::pointCloudEdge(
    pcl::PointCloud<PointT>::Ptr cloud,
    const cv::Mat &image, const cv::Mat &edge_img,
    const int contour_thresh) {
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
          for (int j = 0; j < contours[i].size(); j++) {
             cv::circle(cont_img, contours[i][j], 1,
                        cv::Scalar(255, 0, 0), -1);
          }
       }
    }
    std::vector<std::vector<EdgeNormalDirectionPoint> > normal_points;
    std::vector<std::vector<cv::Point> > tangents;
    this->computeEdgeCurvature(
       cont_img, contours, tangents, normal_points);
    
    cv::imshow("Contours", cont_img);
    cv::waitKey(3);

    pcl::PointCloud<pcl::Normal>::Ptr normals(
       new pcl::PointCloud<pcl::Normal>);
    this->estimatePointCloudNormals(cloud, normals, 0.03f, false);
    pcl::PointCloud<PointT>::Ptr concave_cloud(new pcl::PointCloud<PointT>);
    for (int j = 0; j < normal_points.size(); j++) {
       for (int i = 0; i < normal_points[j].size(); i++) {
          EdgeNormalDirectionPoint point_info = normal_points[j][i];
          cv::Point2f n_pt1 = point_info.normal_pt1;
          cv::Point2f n_pt2 = point_info.normal_pt2;
          cv::Point2f e_pt = (n_pt1 + n_pt2);
          e_pt = cv::Point2f(e_pt.x/2, e_pt.y/2);
          int ept_index = e_pt.x + (e_pt.y * image.cols);
          int pt1_index = n_pt1.x + (n_pt1.y * image.cols);
          int pt2_index = n_pt2.x + (n_pt2.y * image.cols);

          // std::cout << "Index: " << ept_index << "\t "
          //           << pt1_index << "\t" << pt2_index<< std::endl;
          if (pt1_index > -1 && pt2_index > -1 &&  ept_index > -1 &&
             pt1_index <= 307200 && pt2_index <= 307200 &&
              ept_index <= 307200) {
             pcl::Normal n1 = normals->points[pt1_index];
             pcl::Normal n2 = normals->points[pt2_index];
             Eigen::Vector3f n1_vec = Eigen::Vector3f(
                n1.normal_x, n1.normal_y, n1.normal_z);
             Eigen::Vector3f n2_vec = Eigen::Vector3f(
                n2.normal_x, n2.normal_y, n2.normal_z);
             float cos_theta = n1_vec.dot(n2_vec) /
                (n1_vec.norm() * n2_vec.norm());
             float angle = std::acos(cos_theta);
             if (angle < CV_PI/2) {
                PointT pt = cloud->points[ept_index];
                pt.r = 255;
                pt.b = 0;
                pt.g = 0;
                concave_cloud->push_back(pt);
             }
          }
       }
    }
    cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*concave_cloud, *cloud);
}

void InteractiveSegmentation::computeEdgeCurvature(
    const cv::Mat &image,
    const std::vector<std::vector<cv::Point> > &contours,
    std::vector<std::vector<cv::Point> > &tangents,
    std::vector<std::vector<EdgeNormalDirectionPoint> > &normal_points) {
    if (contours.empty()) {
       ROS_ERROR("-- no contours found");
       return;
    }
    normal_points.clear();
    cv::Mat img = image.clone();
    for (int j = 0; j < contours.size(); j++) {
       std::vector<cv::Point> tangent;
       std::vector<float> t_gradient;
       std::vector<EdgeNormalDirectionPoint> norm_tangt;
       cv::Point2f edge_pt = contours[j].front();
       cv::Point2f edge_tngt = contours[j].back() - contours[j][1];
       tangent.push_back(edge_tngt);
       float grad = (edge_tngt.y - edge_pt.y) / (edge_tngt.x - edge_pt.x);
       t_gradient.push_back(grad);
       const int neighbor_pts = 0;
       if (contours[j].size() > 2) {
          for (int i = sizeof(char) + neighbor_pts;
               i < contours[j].size() - sizeof(char) - neighbor_pts;
               i++) {
             edge_pt = contours[j][i];
             edge_tngt = contours[j][i-1-neighbor_pts] -
                contours[j][i+1+neighbor_pts];
            tangent.push_back(edge_tngt);
            grad = (edge_tngt.y - edge_pt.y) / (edge_tngt.x - edge_pt.x);
            t_gradient.push_back(grad);
            cv::Point2f pt1 = edge_tngt + edge_pt;
            cv::Point2f trans = pt1 - edge_pt;
            cv::Point2f ortho_pt1 = edge_pt + cv::Point2f(-trans.y, trans.x);
            cv::Point2f ortho_pt2 = edge_pt - cv::Point2f(-trans.y, trans.x);
            
            norm_tangt.push_back(EdgeNormalDirectionPoint(
                                       ortho_pt1, ortho_pt2,
                                       edge_pt - edge_tngt,
                                       edge_pt + edge_tngt));
            
            cv::line(img, ortho_pt1, ortho_pt2, cv::Scalar(0, 255, 0), 1);
            cv::line(img, edge_pt + edge_tngt, edge_pt -  edge_tngt,
                     cv::Scalar(255, 0, 255), 1);
          }
      }
       tangents.push_back(tangent);
       normal_points.push_back(norm_tangt);
    }
    cv::imshow("tangent", img);
}

template<class T>
void InteractiveSegmentation::estimatePointCloudNormals(
    const pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<pcl::Normal>::Ptr normals,
    const T k, bool use_knn) const {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: The Input cloud is Empty.....");
       return;
    }
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setNumberOfThreads(8);
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
    ne.setSearchMethod(tree);
    if (use_knn) {
        ne.setKSearch(k);
    } else {
        ne.setRadiusSearch(k);
    }
    ne.compute(*normals);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "interactive_segmentation");
    InteractiveSegmentation is;
    ros::spin();
    return 0;
}
