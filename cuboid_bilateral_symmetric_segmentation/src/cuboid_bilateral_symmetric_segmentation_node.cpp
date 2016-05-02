
#include <cuboid_bilateral_symmetric_segmentation/cuboid_bilateral_symmetric_segmentation.h>

CuboidBilateralSymmetricSegmentation::CuboidBilateralSymmetricSegmentation() {
   
    this->onInit();
}

void CuboidBilateralSymmetricSegmentation::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
      "cloud", 1);
    this->pub_edge_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "cloud2", 1);
     this->pub_indices_ = this->pnh_.advertise<
        jsk_msgs::ClusterPointIndices>("indices", 1);
}

void CuboidBilateralSymmetricSegmentation::subscribe() {
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_normal_.subscribe(this->pnh_, "input_normals", 1);
    this->sub_indices_.subscribe(this->pnh_, "input_indices", 1);
    this->sub_boxes_.subscribe(this->pnh_, "input_boxes", 1);
    
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
                                       SyncPolicy> >(100);
    this->sync_->connectInput(this->sub_cloud_,
                              this->sub_indices_,
                              // this->sub_normal_,
                              this->sub_boxes_);
    this->sync_->registerCallback(
        boost::bind(&CuboidBilateralSymmetricSegmentation::cloudCB,
                    this, _1, _2, _3));
}

void CuboidBilateralSymmetricSegmentation::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_normal_.unsubscribe();
    this->sub_boxes_.unsubscribe();
    this->sub_indices_.unsubscribe();
}

void CuboidBilateralSymmetricSegmentation::cloudCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const jsk_msgs::ClusterPointIndices::ConstPtr &indices_msg,
    const jsk_msgs::BoundingBoxArray::ConstPtr &box_msg) {
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    std::vector<pcl::PointIndices::Ptr> indices =
       pcl_conversions::convertToPCLPointIndices(indices_msg->cluster_indices);
    if (indices.size() != box_msg->boxes.size()) {
       ROS_ERROR("INDICES AND BOUNDING BOX ARRAY SIZE NOT EQUAL");
       return;
    }

    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

void boundingBoxSymmetricalAxisPlane(
    std::vector<pcl::ModelCoefficients::Ptr> &plane_coefficients,
    const jsk_msgs::BoundingBoxArray::ConstPtr &box_msg) {
    plane_coefficients.clear();
    
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "cuboid_bilateral_symmetric_segmentation");
    CuboidBilateralSymmetricSegmentation cbss;
    ros::spin();
    return 0;
}

