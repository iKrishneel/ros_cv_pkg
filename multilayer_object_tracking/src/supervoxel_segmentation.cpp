
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

void SupervoxelSegmentation::configCallback(
    Config &config, uint32_t level) {
    boost::mutex::scoped_lock lock(mutex_);
    color_importance_ = config.color_importance;
    spatial_importance_ = config.spatial_importance;
    normal_importance_ = config.normal_importance;
    voxel_resolution_ = config.voxel_resolution;
    seed_resolution_ = config.seed_resolution;
    use_transform_ = config.use_transform;
}
