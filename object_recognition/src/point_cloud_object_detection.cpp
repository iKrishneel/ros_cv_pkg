
#include <object_recognition/point_cloud_object_detection.h>
#include <vector>
#include <iostream>

PointCloudObjectDetection::PointCloudObjectDetection() {

    filter_indices_ = pcl::PointIndices::Ptr(new pcl::PointIndices);
    this->subscribe();
    
    this->pub_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>(
       "/pointcloud_indices/output/cloud", 1);
}

void PointCloudObjectDetection::subscribe() {
   
    // this->sub_rects_ = this->pnh_.subscribe(
    //    "/object_detection/output/rects", sizeof(char),
    //    &PointCloudObjectDetection::jskRectArrayCb, this);

    this->sub_cloud_ = this->pnh_.subscribe(
       "/camera/depth_registered/points"
       /*"/plane_extraction/output_nonplane_cloud"*/, sizeof(char),
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

    std::cout << "Cloud: " << cloud->size() << std::endl;
    
    //if (!this->filter_indices_->indices.empty()) {
       // pcl::ExtractIndices<PointT>::Ptr eifilter(
       // new pcl::ExtractIndices<PointT>);
       // eifilter->setInputCloud(cloud);
       // eifilter->setIndices(filter_indices_);
       // eifilter->filter(*cloud);
       sensor_msgs::PointCloud2 ros_cloud;
       pcl::toROSMsg(*cloud, ros_cloud);
       ros_cloud.header = cloud_msg->header;
       pub_cloud_.publish(ros_cloud);
       //}
}

void PointCloudObjectDetection::jskRectArrayCb(
    const jsk_recognition_msgs::RectArray &rect_msg) {
   
    cloud_width = 640;
    cloud_height = 480;
    filter_indices_ = pcl::PointIndices::Ptr(new pcl::PointIndices);
    for (int i = 0; i < rect_msg.rects.size(); i++) {
       jsk_recognition_msgs::Rect rect = rect_msg.rects[i];
       pcl::PointIndices::Ptr pt_indices(new pcl::PointIndices);
       this->extractPointCloudIndicesFromJSKRect(rect, filter_indices_);
    }
}


// pcl::PointIndices::Ptr
void PointCloudObjectDetection::extractPointCloudIndicesFromJSKRect(
    jsk_recognition_msgs::Rect rect, pcl::PointIndices::Ptr pt_indices) {
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
    // pcl::PointIndices::Ptr pt_indices(new pcl::PointIndices);
    for (int j = rect.y; j < (rect.y + rect.height); j++) {
       for (int i = rect.x; i < (rect.x + rect.width); i++) {
          int index = i + (j * cloud_width);
          pt_indices->indices.push_back(index);
       }
    }
    // return pt_indices;
}
