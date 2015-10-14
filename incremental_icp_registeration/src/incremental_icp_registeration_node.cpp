
#include <incremental_icp_registeration/incremental_icp_registeration.h>

IncrementalICPRegisteration::IncrementalICPRegisteration() :
    set_init(true) {
    this->reg_cloud = pcl::PointCloud<PointT>::Ptr(
       new pcl::PointCloud<PointT>);
    this->prev_cloud = pcl::PointCloud<PointT>::Ptr(
        new pcl::PointCloud<PointT>);
    this->global_transformation = Eigen::Matrix4f::Identity();
    this->onInit();
}

void IncrementalICPRegisteration::onInit() {
    this->subscribe();
    this->pub_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>(
       "/incremental_icp_registeration/output/cloud", sizeof(char));
    this->pub_regis_ = nh_.advertise<sensor_msgs::PointCloud2>(
       "/incremental_icp_registeration/output/registered", sizeof(char));
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

    if (!cloud->empty()) {
      std::vector<int> index;
      pcl::removeNaNFromPointCloud<PointT>(*cloud, *cloud, index);
    }
    pcl::PointCloud<PointT>::Ptr orig_cloud(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(*cloud, *orig_cloud);
    
    if (!this->prev_cloud->empty()) {
       bool downsample = true;
       const float leaf_size = 0.2f;
       pcl::PointCloud<PointT>::Ptr source(new pcl::PointCloud<PointT>);
       pcl::PointCloud<PointT>::Ptr target(new pcl::PointCloud<PointT>);
       if (downsample) {
         this->downsampleCloud(this->prev_cloud, source, leaf_size);
         this->downsampleCloud(cloud, target, leaf_size);
       } else {
          pcl::copyPointCloud<PointT, PointT>(*prev_cloud, *source);
          pcl::copyPointCloud<PointT, PointT>(*cloud, *target);
       }
       Eigen::Matrix4f transformation;
       pcl::PointCloud<PointT>::Ptr output(new pcl::PointCloud<PointT>);
       if (this->icpAlignPointCloud(this->prev_cloud, cloud, output, transformation)) {
         ROS_INFO("\033[34m CLOUD ALIGHNED \033[0m");
         pcl::transformPointCloud(*orig_cloud, *output, transformation);
         *output += *reg_cloud;
         this->global_transformation = global_transformation * transformation;
         reg_cloud->clear();
         pcl::copyPointCloud<PointT, PointT>(*output, *reg_cloud);
         pcl::copyPointCloud<PointT, PointT>(*orig_cloud, *prev_cloud);
         
         std::cout << "Transform: " << transformation  << "\n";
       }
    } else {
       pcl::copyPointCloud<PointT, PointT>(*cloud, *prev_cloud);
       ROS_INFO("\033[34m INITIAL MODEL SET\033[0m");
       set_init = false;
    }

    // pcl::copyPointCloud<PointT, PointT>(*cloud, *reg_cloud);
    // this->downsampleCloud(this->reg_cloud, reg_cloud, 0.1f);
    // std::cout <<"Size: " << reg_cloud->size()  << "\n";
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);

    sensor_msgs::PointCloud2 ros_regis;
    pcl::toROSMsg(*reg_cloud, ros_regis);
    ros_regis.header = cloud_msg->header;
    this->pub_regis_.publish(ros_regis);
}

bool IncrementalICPRegisteration::icpAlignPointCloud(
    const pcl::PointCloud<PointT>::Ptr source,
    const pcl::PointCloud<PointT>::Ptr target,
    pcl::PointCloud<PointT>::Ptr output, Eigen::Matrix4f &final_transform) {
    if (source->empty() || target->empty()) {
       ROS_ERROR("THE INPUT PAIRS ARE EMPTY");
       return false;
    }
    ROS_INFO("\033[34m RUNNING ICP ALIGNMENT \033[0m");
    // pcl::PointCloud<pcl::PointNormal>::Ptr n_source(
    //    new pcl::PointCloud<pcl::PointNormal>);
    // pcl::PointCloud<pcl::PointNormal>::Ptr n_target(
    //     new pcl::PointCloud<pcl::PointNormal>);
    // this->estimateNormal(source, n_source);
    // this->estimateNormal(target, n_target);

    // ICPPointRepresentation icp_point;
    // float alpha[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    // icp_point.setRescaleValues(alpha);
    
    // pcl::IterativeClosestPointNonLinear<pcl::PointNormal, pcl::PointNormal> reg;
    pcl::IterativeClosestPointNonLinear<PointT, PointT> reg;
    reg.setTransformationEpsilon(1e-6);
    reg.setMaxCorrespondenceDistance(0.1f);
    // reg.setPointRepresentation(boost::make_shared<
    //                            const ICPPointRepresentation> (icp_point));
    // reg.setInputSource(n_source);
    // reg.setInputTarget(n_target);

    reg.setInputSource(source);
    reg.setInputTarget(target);
    reg.setMaximumIterations(5);
    reg.setEuclideanFitnessEpsilon(0.01);
    reg.setRANSACIterations(100);
    reg.setRANSACOutlierRejectionThreshold(0.05);
    
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f prev;
    // pcl::PointCloud<pcl::PointNormal>::Ptr reg_result = n_source;


    ROS_WARN("STARTING REGISTRATION");

    pcl::PointCloud<PointT>::Ptr tmp(new pcl::PointCloud<PointT>);
    reg.align(*tmp);
    
    final_transform = reg.getFinalTransformation().inverse();
    
    
    // const int max_iter = 2;
    // for (int i = 0; i < max_iter; i++) {
    //    n_source = reg_result;
    //    reg.setInputSource(n_source);
    //    reg.align(*reg_result);
    //    trans = reg.getFinalTransformation() * trans;

    //    if (fabs((reg.getLastIncrementalTransformation() - prev).sum()) <
    //        reg.getTransformationEpsilon()) {
    //      reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001f);
    //    }
    //    prev = reg.getLastIncrementalTransformation();
    // }

    ROS_WARN("FINISHED REGISTRATION");
        
    // Eigen::Matrix4f tgt2src = trans.inverse();
    // pcl::transformPointCloud(*target, *output, tgt2src);
    // final_transform = tgt2src;
    // *output += *source;
    return true;
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
