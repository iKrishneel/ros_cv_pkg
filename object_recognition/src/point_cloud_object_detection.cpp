
#include <object_recognition/point_cloud_object_detection.h>
#include <vector>

PointCloudObjectDetection::PointCloudObjectDetection() {

    this->subscribe();
    this->pub_indices_ = this->pnh_.advertise<
      jsk_recognition_msgs::ClusterPointIndices>(
         "/object_detection/output/indices", 1);
    this->pub_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>(
       "/object_detection/output/cloud", 1);
}

void PointCloudObjectDetection::subscribe() {

    this->sub_rects_ = this->pnh_.subscribe(
      "/object_detection/output/rects", sizeof(char),
      &PointCloudObjectDetection::jskRectArrayCb, this);
    this->sub_cloud_ = this->pnh_.subscribe(
       "/camera/depth_registered/points", sizeof(char),
       &PointCloudObjectDetection::cloudCallback, this);
}

void PointCloudObjectDetection::unsubscribe() {
    ROS_INFO("Unsubscribing from ROS Topics...");
    this->sub_cloud_.shutdown();
}

void PointCloudObjectDetection::cloudCallback(
    const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    boost::mutex::scoped_lock lock(this->mutex_);
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // pcl::PointCloud<PointT>::Ptr output(new pcl::PointCloud<PointT>);
    cloud_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(*cloud, *cloud_);
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    // cloud_width = cloud->width;
    
    
    jsk_recognition_msgs::ClusterPointIndices ros_indices;
    ros_indices.cluster_indices = pcl_conversions::convertToROSPointIndices(
       rect_cluster_indices, cloud_msg->header);
    ros_indices.header = cloud_msg->header;
    pub_indices_.publish(ros_indices);
    pub_cloud_.publish(ros_cloud);
}

void PointCloudObjectDetection::jskRectArrayCb(
    const jsk_recognition_msgs::RectArray &rect_msg) {
   // std::vector<pcl::PointIndices> rect_cluster_indices;
    cloud_width = 640;
    cloud_height = 480;
    rect_cluster_indices.clear();

    pcl::ExtractIndices<PointT>::Ptr eifilter(
       new pcl::ExtractIndices<PointT>);
    eifilter->setInputCloud(this->cloud_);
    
    for (int i = 0; i < rect_msg.rects.size(); i++) {
       jsk_recognition_msgs::Rect rect = rect_msg.rects[i];
       pcl::PointIndices::Ptr pt_indices(new pcl::PointIndices);
       pt_indices = this->extractPointCloudIndicesFromJSKRect(rect);
       if (!pt_indices->indices.empty()) {
          rect_cluster_indices.push_back(*pt_indices);

          eifilter->setIndices(pt_indices);
          pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
          eifilter->filter(*tmp_cloud);
       }
    }
    
    
    /*
    jsk_recognition_msgs::ClusterPointIndices ros_indices;
    ros_indices.cluster_indices = pcl_conversions::convertToROSPointIndices(
       rect_cluster_indices, rect_msg.header);
    ros_indices.header = rect_msg.header;
    pub_indices_.publish(ros_indices);
    */
}

pcl::PointIndices::Ptr
PointCloudObjectDetection::extractPointCloudIndicesFromJSKRect(
    jsk_recognition_msgs::Rect rect) {
    if (rect.x < 0) {
       rect.x = 0;
    }
    if (rect.y < 0) {
       rect.y = 0;
    }
    if (rect.x + rect.width > cloud_width) {
       rect.width -= ((rect.x + rect.width) - cloud_width);
    }
    if (rect.y + rect.height > cloud_height) {
       rect.height -= ((rect.y + rect.height) - cloud_height);
    }
    if (rect.width == 0 || rect.height == 0) {
       // return pcl::PointIndices();
    }
    pcl::PointIndices::Ptr pt_indices;
    for (int j = rect.y; j < (rect.y + rect.height); j++) {
       for (int i = rect.x; i < (rect.x + rect.width); i++) {
          int index = i + (j * cloud_width);
          pt_indices->indices.push_back(index);
       }
    }
    return pt_indices;
}


/**
 * service all to Euclidean Clustering
 */
void PointCloudObjectDetection::euclideanClusteringServiceHandler(
    pcl::PointCloud<PointT>::Ptr cloud) {
    sensor_msgs::PointCloud2 cloud_msg;
    ros::ServiceClient client = pnh_.serviceClient<
       jsk_pcl_ros::EuclideanSegment>("euclidean_clustering");
    jsk_pcl_ros::EuclideanSegment es_srv;
    es_srv.request.input = cloud_msg;
    
}


void PointCloudObjectDetection::euclideanClusteringService(
    jsk_pcl_ros::EuclideanSegment::Request &req,
    jsk_pcl_ros::EuclideanSegment::Response &res) {
   
}


/**
 * 
 */
/*
pcl::PointCloud<PointT>::Ptr
PointCloudObjectDetection::extractPointCloudFromJSKRect(
    const pcl::PointCloud<PointT>::Ptr cloud,
    jsk_recognition_msgs::Rect rect) {
    if (rect.x < 0) {
       rect.x = 0;
    }
    if (rect.y < 0) {
       rect.y = 0;
    }
    if (rect.x + rect.width > cloud_width) {
       rect.width -= ((rect.x + rect.width) - cloud_width);
    }
    if (rect.y + rect.height > cloud_height) {
       rect.height -= ((rect.y + rect.height) - cloud_height);
    }
    if (rect.width == 0 || rect.height == 0) {
       // return pcl::PointIndices();
    }
    pcl::PointCloud<PointT>::Ptr ncloud(new pcl::PointCloud<PointT>);
    for (int j = rect.y; j < (rect.y + rect.height); j++) {
       for (int i = rect.x; i < (rect.x + rect.width); i++) {
          int index = i + (j * cloud_width);
          ncloud.push_back(cloud->points[index]);
       }
    }
    return ncloud;
}
*/
