// Copyright (C) 2016 by Krishneel Chaudhary, JSK Lab,a
// The University of Tokyo, Japan

#include <interactive_segmentation/object_region_estimation.h>

ObjectRegionEstimation::ObjectRegionEstimation() :
    num_threads_(8) {
    counter_ = 0;
    this->prev_cloud_ = pcl::PointCloud<PointT>::Ptr(
       new pcl::PointCloud<PointT>);
    this->onInit();
}

void ObjectRegionEstimation::onInit() {

    this->srv_client_ = this->pnh_.serviceClient<
       interactive_segmentation::Feature3DClustering>(
          "feature3d_clustering_srv");
  
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/object_region_estimation/output/cloud", 1);
    this->pub_indices_ = this->pnh_.advertise<
       jsk_recognition_msgs::ClusterPointIndices>(
          "/object_region_estimation/output/indices", 1);
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
    this->header_ = cloud_msg->header;
    
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
    // pcl::PointCloud<PointI>::Ptr keypoints(new pcl::PointCloud<PointI>);
    // this->keypoints3D(keypoints, cloud);

    // pcl::PointCloud<SHOT352>::Ptr descriptors(new pcl::PointCloud<SHOT352>);
    // this->features3D(descriptors, cloud, normals, keypoints);
    /*
    std::vector<pcl::PointIndices> all_indices;
    this->clusterFeatures(all_indices, cloud, normals, 5, 0.5);

    std::cout << "Cluster Size: " << all_indices.size() << "\n";
    
    jsk_recognition_msgs::ClusterPointIndices ros_indices;
    ros_indices.cluster_indices = pcl_conversions::convertToROSPointIndices(
        all_indices, cloud_msg->header);
    ros_indices.header = cloud_msg->header;
    pub_indices_.publish(ros_indices);
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    // this->pub_cloud_.publish(ros_cloud);
    */

    this->stableVariation(tmp_cloud);

    if (counter_++ == 0) {
       *prev_cloud_ = *tmp_cloud;
    }
    
    // this->prev_cloud_->clear();
    // pcl::copyPointCloud<PointT, PointT>(*tmp_cloud, *prev_cloud_);
    
    
    std::cout << "GO TO SLEEP: " << prev_cloud_->size()  << "\n";
    // ros::Duration(5).sleep();
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

void ObjectRegionEstimation::clusterFeatures(
    std::vector<pcl::PointIndices> &all_indices,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<Normal>::Ptr descriptors,
    const int min_size, const float max_distance) {
    if (descriptors->empty()) {
      ROS_ERROR("ERROR: EMPTY FEATURES.. SKIPPING CLUSTER SRV");
      return;
    }
    interactive_segmentation::Feature3DClustering srv;
    for (int i = 0; i < descriptors->size(); i++) {
      jsk_recognition_msgs::Histogram hist;
      hist.histogram.push_back(cloud->points[i].x);
      hist.histogram.push_back(cloud->points[i].y);
      hist.histogram.push_back(cloud->points[i].z);
      hist.histogram.push_back(descriptors->points[i].normal_x);
      hist.histogram.push_back(descriptors->points[i].normal_y);
      hist.histogram.push_back(descriptors->points[i].normal_z);
      srv.request.features.push_back(hist);
    }
    srv.request.min_samples = min_size;
    srv.request.max_distance = max_distance;
    if (this->srv_client_.call(srv)) {
       int max_label = srv.response.argmax_label;
       if (max_label == -1) {
          return;
       }
       all_indices.clear();
       all_indices.resize(max_label + 1);      
       for (int i = 0; i < srv.response.labels.size(); i++) {
          int index = srv.response.labels[i];
          if (index > -1) {
             all_indices[index].indices.push_back(i);
          }
       }
    } else {
       ROS_ERROR("SRV CLIENT CALL FAILED");
       return;
    }
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

void ObjectRegionEstimation::stableVariation(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const float resolution) {
    pcl::octree::OctreePointCloudChangeDetector<PointT> octree(0.10f);
    octree.setInputCloud(this->prev_cloud_);
    octree.addPointsFromInputCloud();
    octree.switchBuffers();
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    octree.getPointIndicesFromNewVoxels(indices->indices);
    pcl::PointCloud<PointT>::Ptr change(new pcl::PointCloud<PointT>);
    for (int i = 0; i < indices->indices.size(); i++) {
       int ind = indices->indices[i];
       change->push_back(cloud->points[ind]);
    }
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*change, ros_cloud);
    ros_cloud.header = header_;
    this->pub_cloud_.publish(ros_cloud);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "object_region_estimation");
    ObjectRegionEstimation ore;
    ros::spin();
    return 0;
}
