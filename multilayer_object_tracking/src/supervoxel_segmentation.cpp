
#include <multilayer_object_tracking/supervoxel_segmentation.h>
#include <map>

SupervoxelSegmentation::SupervoxelSegmentation() {
    srv_ = boost::shared_ptr<dynamic_reconfigure::Server<Config> >(
       new dynamic_reconfigure::Server<Config>);
    dynamic_reconfigure::Server<Config>::CallbackType f =
       boost::bind(
          &SupervoxelSegmentation::configCallback, this, _1, _2);
    srv_->setCallback(f);
}

void SupervoxelSegmentation::supervoxelSegmentation(
    const pcl::PointCloud<PointT>::Ptr cloud,
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > &supervoxel_clusters,
    std::multimap<uint32_t, uint32_t> &supervoxel_adjacency) {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: Supervoxel input cloud empty...");
       return;
    }
    boost::mutex::scoped_lock lock(mutex_);
    pcl::SupervoxelClustering<PointT> super(voxel_resolution_,
                                            seed_resolution_,
                                            use_transform_);
    super.setInputCloud(cloud);
    super.setColorImportance(color_importance_);
    super.setSpatialImportance(spatial_importance_);
    super.setNormalImportance(normal_importance_);
    supervoxel_clusters.clear();
    super.extract(supervoxel_clusters);
    super.getSupervoxelAdjacency(supervoxel_adjacency);
}

void SupervoxelSegmentation::publishSupervoxel(
    const std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters,
    sensor_msgs::PointCloud2 &ros_cloud,
    jsk_recognition_msgs::ClusterPointIndices &ros_indices,
    const std_msgs::Header &header) {
    pcl::PointCloud<PointT>::Ptr output (new pcl::PointCloud<PointT>);
    std::vector<pcl::PointIndices> all_indices;
    for (std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr >::const_iterator
           it = supervoxel_clusters.begin();
         it != supervoxel_clusters.end();
         ++it) {
      pcl::Supervoxel<PointT>::Ptr super_voxel = it->second;
      pcl::PointCloud<PointT>::Ptr super_voxel_cloud = super_voxel->voxels_;
      pcl::PointIndices indices;
      for (size_t i = 0; i < super_voxel_cloud->size(); i++) {
        indices.indices.push_back(i + output->points.size());
      }
      all_indices.push_back(indices);
      *output = *output + *super_voxel_cloud;
    }
    ros_indices.cluster_indices.clear();
    ros_indices.cluster_indices = this->convertToROSPointIndices(
       all_indices, header);
    ros_cloud.data.clear();
    pcl::toROSMsg(*output, ros_cloud);
    ros_indices.header = header;
    ros_cloud.header = header;
}

void SupervoxelSegmentation::targetDescriptiveSurfelsIndices(
    const jsk_recognition_msgs::ClusterPointIndices &sv_indices,
    const std::vector<uint32_t> &tdp_list,
    jsk_recognition_msgs::ClusterPointIndices &ros_indices) {
    ros_indices.cluster_indices.clear();
    for (std::vector<uint32_t>::const_iterator it = tdp_list.begin();
         it != tdp_list.end(); it++) {
       ros_indices.cluster_indices.push_back(sv_indices.cluster_indices[*it]);
    }
    ros_indices.header = sv_indices.header;
}

std::vector<pcl_msgs::PointIndices>
SupervoxelSegmentation::convertToROSPointIndices(
    const std::vector<pcl::PointIndices> cluster_indices,
    const std_msgs::Header& header) {
    std::vector<pcl_msgs::PointIndices> ret;
    for (size_t i = 0; i < cluster_indices.size(); i++) {
       pcl_msgs::PointIndices ros_msg;
       ros_msg.header = header;
       ros_msg.indices = cluster_indices[i].indices;
       ret.push_back(ros_msg);
    }
    return ret;
}

void SupervoxelSegmentation::configCallback(
    Config &config, uint32_t level) {
    boost::mutex::scoped_lock lock(mutex_);
    this->color_importance_ = config.color_importance;
    this->spatial_importance_ = config.spatial_importance;
    this->normal_importance_ = config.normal_importance;
    this->voxel_resolution_ = config.voxel_resolution;
    this->seed_resolution_ = config.seed_resolution;
    this->use_transform_ = config.use_transform;

    this->min_cluster_size_ = static_cast<int>(config.min_cluster_size);
    this->threshold_ = static_cast<float>(config.threshold);
    this->bin_size_ = static_cast<int>(config.bin_size);
    this->eps_distance_ = static_cast<float>(config.eps_distance);
    this->eps_min_samples_ = static_cast<float>(config.eps_min_samples);
    this->vfh_scaling_ = static_cast<float>(config.vfh_scaling);
    this->color_scaling_ = static_cast<float>(config.color_scaling);
    this->update_tracker_reference_ = config.update_tracker_reference;
}
