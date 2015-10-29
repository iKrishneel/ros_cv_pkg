#include <hierarchical_object_learning/hierarchical_object_detection.h>

HierarchicalObjectDetection::HierarchicalObjectDetection() {
  
}

void HierarchicalObjectDetection::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/hierarchical_object_detection/output/cloud", 1);
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/hierarchical_object_detection/output/image", 1);
    this->pub_pose_ = this->pnh_.advertise<geometry_msgs::PoseStamped>(
       "/hierarchical_object_detection/output/pose", 1);
}

void HierarchicalObjectDetection::subscribe() {
       this->sub_info_.subscribe(this->pnh_, "input_info", 1);
       this->sub_image_.subscribe(this->pnh_, "input_image", 1);
       this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
       this->sync_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy> >(100);
       sync_->connectInput(sub_info_, sub_image_, sub_cloud_);
       sync_->registerCallback(boost::bind(
                                  &HierarchicalObjectDetection::callback,
                                  this, _1, _2, _3));
}

void HierarchicalObjectDetection::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_image_.unsubscribe();
}


void HierarchicalObjectDetection::callback(
    const sensor_msgs::CameraInfo::ConstPtr &info_msg,
    const sensor_msgs::Image::ConstPtr &image_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
  
    // cv::Mat image = cv_bridge::toCvShare(
    //    image_msg, image_msg->encoding)->image;
    // pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    // pcl::fromROSMsg(*cloud_msg, *cloud);

    // pcl::PointCloud<pcl::Normal>::Ptr normals(
    // new pcl::PointCloud<pcl::Normal>);
    // this->estimatePointCloudNormals<float>(cloud, normals, 16, false);

    // cv::Mat featureMD;
    // this->pointFeaturesBOWDescriptor(
    //    cloud, normals, featureMD, this->cluster_size_);

    
    // cv_bridge::CvImage pub_img(
    //     image_msg->header, sensor_msgs::image_encodings::BGR8, image);
    // sensor_msgs::PointCloud2 ros_cloud;
    // pcl::toROSMsg(*cloud, ros_cloud);
    // ros_cloud.header = cloud_msg->header;   

    // this->pub_cloud_.publish(ros_cloud);
    // this->pub_image_.publish(pub_img.toImageMsg());
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "hierarchical_object_detection");
    HierarchicalObjectDetection hod;
    ros::spin();
    return 0;
}
