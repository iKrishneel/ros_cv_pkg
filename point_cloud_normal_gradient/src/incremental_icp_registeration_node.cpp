
// #include <incremental_icp_registeration/incremental_icp_registeration.h>
#include <point_cloud_normal_gradient/incremental_icp_registeration.h>

IncrementalICPRegisteration::IncrementalICPRegisteration() :
    set_init(true) {
    this->initial_cloud = pcl::PointCloud<PointT>::Ptr(
       new pcl::PointCloud<PointT>);
    this->onInit();
}

void IncrementalICPRegisteration::onInit() {
    this->subscribe();
    this->pub_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>(
       "/incremental_icp_registeration/output/cloud", sizeof(char));
}

void IncrementalICPRegisteration::subscribe() {
    this->sub_cloud_ = nh_.subscribe("input", 1,
       &IncrementalICPRegisteration::callback, this);
}

void IncrementalICPRegisteration::unsubscribe() {
    this->sub_cloud_.shutdown();
}

void IncrementalICPRegisteration::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if (!this->initial_cloud->empty() && !set_init) {
       bool downsample = false;
       pcl::PointCloud<PointT>::Ptr source(new pcl::PointCloud<PointT>);
       pcl::PointCloud<PointT>::Ptr target(new pcl::PointCloud<PointT>);
       if (downsample) {
          this->downsampleCloud(this->initial_cloud, source);
          this->downsampleCloud(cloud, target);
       } else {
          pcl::copyPointCloud<PointT, PointT>(*initial_cloud, *source);
          pcl::copyPointCloud<PointT, PointT>(*cloud, *target);
       }
    }
    
    if (set_init) {
       pcl::copyPointCloud<PointT, PointT>(*cloud, *initial_cloud);
       ROS_INFO("\033[34m INITIAL MODEL SET\033[0m");
       set_init = false;
    }
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

bool IncrementalICPRegisteration::icpAlignPointCloud(
    const pcl::PointCloud<PointT>::Ptr source,
    const pcl::PointCloud<PointT>::Ptr target,
    pcl::PointCloud<PointT>::Ptr output, Eigen::Matrix4f &final_transform) {
    if (source->empty() || target->empty()) {
       ROS_ERROR("THE INPUT PAIRS ARE EMPTY");
       return false;
    }
    pcl::PointCloud<pcl::PointNormal>::Ptr n_source(
       new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr n_target(
       new pcl::PointCloud<pcl::PointNormal>);

    this->estimateNormal(source, n_source);
    this->estimateNormal(target, n_target);


    ICPPointRepresentation icp_point;
    float alpha[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    icp_point.setRescaleValues(alpha);
    
    pcl::IterativeClosestPointNonLinear<pcl::PointNormal, pcl::PointNormal> reg;
    reg.setTransformationEpsilon(1e-6);
    reg.setMaxCorrespondenceDistance(0.05f);
    reg.setPointRepresentation(boost::make_shared<
                               const ICPPointRepresentation> (icp_point));
    reg.setInputSource(n_source);
    reg.setInputTarget(n_target);

    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f prev;
    Eigen::Matrix4f tgt2src;
    pcl::PointCloud<pcl::PointNormal>::Ptr reg_result = n_source;
    reg.setMaximumIterations(5);

    const int max_iter = 30;
    for (int i = 0; i < max_iter; i++) {
       n_source = reg_result;
       reg.setInputSource(n_source);
       reg.align(*reg_result);
    }

}

void IncrementalICPRegisteration::downsampleCloud(
    const pcl::PointCloud<PointT>::Ptr input,
    pcl::PointCloud<PointT>::Ptr output, const float leaf_size) {
    pcl::VoxelGrid<PointT> grid;
    grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    grid.setInputCloud(input);
    grid.filter(*output);
}

void IncrementalICPRegisteration::estimateNormal(
    const pcl::PointCloud<PointT>::Ptr input,
    pcl::PointCloud<pcl::PointNormal>::Ptr normal_out,
    const int k) {
    pcl::NormalEstimation<PointT, pcl::PointNormal> norm_est;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    norm_est.setSearchMethod(tree);
    norm_est.setKSearch(k);
    norm_est.setInputCloud(input);
    norm_est.compute(*normal_out);
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "incremental_icp_registeration");
    IncrementalICPRegisteration iicpr;
    ros::spin();
    return 0;
}
