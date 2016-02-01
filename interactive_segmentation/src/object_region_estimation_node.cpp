// Copyright (C) 2016 by Krishneel Chaudhary, JSK Lab,a
// The University of Tokyo, Japan

#include <interactive_segmentation/object_region_estimation.h>

ObjectRegionEstimation::ObjectRegionEstimation() {
    this->onInit();
}

void ObjectRegionEstimation::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/object_region_estimation/output/cloud", 1);
}

void ObjectRegionEstimation::subscribe() {
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_normal_.subscribe(this->pnh_, "input_normals", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    sync_->connectInput(sub_cloud_, sub_normal_);
    sync_->registerCallback(boost::bind(
                               &ObjectRegionEstimation::callback,
                               this, _1, _2));
}

void ObjectRegionEstimation::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_indices_.unsubscribe();
    this->sub_normal_.unsubscribe();
}

void ObjectRegionEstimation::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::PointCloud2::ConstPtr &normal_msg) {
    pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *tmp_cloud);
    pcl::PointCloud<Normal>::Ptr tmp_normals(new pcl::PointCloud<Normal>);
    pcl::fromROSMsg(*normal_msg, *tmp_normals);
    if (tmp_cloud->size() != tmp_normals->size()) {
       ROS_ERROR("INCORRECT INPUT SIZE");
       return;
    }
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<Normal>::Ptr normals(new pcl::PointCloud<Normal>);
    for (int i = 0; i < tmp_cloud->size(); i++) {
       PointT pt = tmp_cloud->points[i];
       Normal nt = tmp_normals->points[i];
       if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z) &&
           !isnan(nt.normal_x) && !isnan(nt.normal_y) && !isnan(nt.normal_z)) {
          cloud->push_back(pt);
          normals->push_back(tmp_normals->points[i]);
       }
    }
    pcl::PointCloud<PointI>::Ptr keypoints(new pcl::PointCloud<PointI>);
    this->keypoints3D(keypoints, cloud);

    pcl::PointCloud<SHOT352>::Ptr descriptors(new pcl::PointCloud<SHOT352>);
    this->features3D(descriptors, cloud, normals, keypoints);

    std::cout << "DESCRIPTOR SIZE: " << descriptors->size()  << ", "
              << keypoints->size() << std::endl;
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*keypoints, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

void ObjectRegionEstimation::keypoints3D(
    pcl::PointCloud<PointI>::Ptr keypoints,
    const pcl::PointCloud<PointT>::Ptr cloud) {
    if (cloud->empty()) {
       return;
    }
    pcl::HarrisKeypoint3D<PointT, PointI>::Ptr detector(
       new pcl::HarrisKeypoint3D<PointT, PointI>);
    detector->setNonMaxSupression(true);
    detector->setInputCloud(cloud);
    detector->setThreshold(1e-6);
    detector->compute(*keypoints);
}

void ObjectRegionEstimation::features3D(
    pcl::PointCloud<SHOT352>::Ptr descriptors,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<Normal>::Ptr normals,
    const pcl::PointCloud<PointI>::Ptr keypoints) {
    pcl::PointCloud<PointT>::Ptr keypoints_xyz(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointI, PointT>(*keypoints, *keypoints_xyz);
    pcl::SHOTEstimationOMP<PointT, Normal, SHOT352> shot;
    shot.setSearchSurface(cloud);
    shot.setInputCloud(keypoints_xyz);
    shot.setInputNormals(normals);
    shot.setRadiusSearch(0.02f);
    shot.compute(*descriptors);
}

void ObjectRegionEstimation::removeStaticKeypoints(
    pcl::PointCloud<PointI>::Ptr prev_keypoints,
    pcl::PointCloud<PointI>::Ptr curr_keypoints,
    const float threshold) {
    pcl::PointCloud<PointI>::Ptr keypoints(new pcl::PointCloud<PointI>);
    const int size = prev_keypoints->size();
    int match_index[size];
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_) shared(match_index)
#endif
    for (int i = 0; i < prev_keypoints->size(); i++) {
       match_index[i] = -1;
       for (int j = 0; j < curr_keypoints->size(); j++) {
          double distance = pcl::distances::l2(
             prev_keypoints->points[i].getVector4fMap(),
             curr_keypoints->points[j].getVector4fMap());
          if (distance < threshold) {
             match_index[i] = j;
          }
       }
    }
    
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "object_region_estimation");
    ObjectRegionEstimation ore;
    ros::spin();
    return 0;
}
