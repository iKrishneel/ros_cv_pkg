
#include <cluster_point_indices_image/cluster_point_indices_image.h>

ClusterPointIndicesToImage::ClusterPointIndicesToImage() {
    this->subscribe();
    this->onInit();
}

void ClusterPointIndicesToImage::onInit() {
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/cluster_point_indices_to_image/output/cloud", 1);
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/cluster_point_indices_to_image/output/image", 1);
}

void ClusterPointIndicesToImage::subscribe() {
    this->sub_select_.subscribe(this->pnh_, "selected_region", 1);
    this->sub_indices_.subscribe(this->pnh_, "input_indices", 1);
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    sync_->connectInput(sub_select_, sub_indices_, sub_cloud_);
    sync_->registerCallback(boost::bind(
                               &ClusterPointIndicesToImage::callback,
                               this, _1, _2, _3));
}

void ClusterPointIndicesToImage::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_indices_.unsubscribe();
}

void ClusterPointIndicesToImage::callback(
    const jsk_recognition_msgs::ClusterPointIndicesConstPtr &indices_mgs,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {

    std::vector<pcl::PointIndices::Ptr> cluster_indices;
    cluster_indices = this->clusterPointIndicesToPointIndices(indices_mgs);

    
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

std::vector<pcl::PointIndices::Ptr>
ClusterPointIndicesToImage::clusterPointIndicesToPointIndices(
    const jsk_recognition_msgs::ClusterPointIndicesConstPtr &indices_mgs) {
    std::vector<pcl::PointIndices::Ptr> ret;
    int icounter = 0;
    for (int i = 0; i < indices_mgs->cluster_indices.size(); i++) {
       std::vector<int> indices = indices_mgs->cluster_indices[i].indices;
       pcl::PointIndices::Ptr pcl_indices (new pcl::PointIndices);
       pcl_indices->indices = indices;
       ret.push_back(pcl_indices);
       icounter += indices.size();
    }
    std::cout << "Size: " << icounter  << std::endl;
    
    return ret;
}

int main(int argc, char *argv[]) {

    ros::init(argc, argv, "cluster_point_indices_image");
    ClusterPointIndicesToImage cpi2i;
    ros::spin();
    return 0;
}
