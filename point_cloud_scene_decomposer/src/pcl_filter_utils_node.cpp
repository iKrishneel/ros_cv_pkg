// Copyright (C) 2015 by Krishneel Chaudhary, JSK

#include <point_cloud_scene_decomposer/pcl_filter_utils.h>

PointCloudFilterUtils::PointCloudFilterUtils() {
   
    this->subsribe();
    this->onInit();
}

void PointCloudFilterUtils::onInit() {
    pub_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>("target", 1);
}

void PointCloudFilterUtils::subsribe() {
    this->sub_cloud_ = this->pnh_.subscribe(
        "input", 1, &PointCloudFilterUtils::cloudCallback, this);
    
    dynamic_reconfigure::Server<
        pcl_filter_utils::PointCloudFilterUtilsConfig>::CallbackType f =
        boost::bind(&PointCloudFilterUtils::configCallback, this, _1, _2);
    server.setCallback(f);
}

void PointCloudFilterUtils::cloudCallback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msgs) {
    boost::shared_ptr<pcl::PCLPointCloud2> cloud(
        new pcl::PCLPointCloud2);
    pcl_conversions::toPCL(*cloud_msgs, *cloud);
    pcl::PCLPointCloud2 *cloud_filtered = new pcl::PCLPointCloud2;
    this->pclDistanceFilter(cloud, *cloud_filtered);

    sensor_msgs::PointCloud2 out_msg;
    pcl_conversions::moveFromPCL(*cloud_filtered, out_msg);
    this->pub_cloud_.publish(out_msg);
}

void PointCloudFilterUtils::pclDistanceFilter(
    const boost::shared_ptr<pcl::PCLPointCloud2> cloud,
    pcl::PCLPointCloud2 &cloud_filtered) {
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
    pcl::PassThrough<pcl::PCLPointCloud2> pass;
    pass.setInputCloud(cloudPtr);
    pass.setFilterFieldName("z");
    pass.setKeepOrganized(true);
    pass.setFilterLimits(this->min_distance_, this->max_distance_);
    pass.filter(cloud_filtered);
}

void PointCloudFilterUtils::configCallback(
    pcl_filter_utils::PointCloudFilterUtilsConfig &config, uint32_t level) {

    boost::mutex::scoped_lock lock(this->lock_);
    this->min_distance_ = static_cast<float>(config.min_distance);
    this->max_distance_ = static_cast<float>(config.max_distance);
}


int main(int argc, char *argv[]) {

    ros::init(argc, argv, "pcl_filter_utils");
    PointCloudFilterUtils pcfu;
    ros::spin();
    return 0;
}
