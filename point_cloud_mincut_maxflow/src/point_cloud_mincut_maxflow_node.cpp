
#include <point_cloud_mincut_maxflow/point_cloud_mincut_maxflow.h>

PointCloudMinCutMaxFlow::PointCloudMinCutMaxFlow() {
    std::cout << "HELLO" << std::endl;
}

void PointCloudMinCutMaxFlow::onInit() {
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/point_cloud_mincut_maxflow/output/cloud", 1);
    this->pub_obj_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/point_cloud_mincut_maxflow/output/selected_probability", 1);
    this->pub_indices_ = this->pnh_.advertise<
       jsk_recognition_msgs::ClusterPointIndices>(
          "/point_cloud_mincut_maxflow/output/indices", 1);
}

void PointCloudMinCutMaxFlow::subscribe() {
       this->sub_mask_.subscribe(this->pnh_, "in_mask", 1);
       this->sub_cloud_.subscribe(this->pnh_, "in_cloud", 1);
       this->sync_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy> >(100);
       sync_->connectInput(sub_cloud_, sub_mask_);
       sync_->registerCallback(boost::bind(&PointCloudMinCutMaxFlow::callback,
                                           this, _1, _2));
}

void PointCloudMinCutMaxFlow::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_mask_.unsubscribe();
}

void PointCloudMinCutMaxFlow::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::PointCloud2::ConstPtr &mask_msg) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    pcl::PointCloud<PointT>::Ptr mask_cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*mask_msg, *mask_cloud);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "point_cloud_mincut_maxflow");
    PointCloudMinCutMaxFlow pcmm;
    ros::spin();
    return 0;
}
