
#include <handheld_object_registration/particle_filter_tracking.h>

ParticleFilters::ParticleFilters():
    tracker_init_(false), downsampling_grid_size_(0.01) {

    std::vector<double> default_step_covariance =
       std::vector<double>(6, 0.015 * 0.015);
    default_step_covariance[3] *= 40.0;
    default_step_covariance[4] *= 40.0;
    default_step_covariance[5] *= 40.0;

    std::vector<double> initial_noise_covariance =
       std::vector<double>(6, 0.00001);
    std::vector<double> default_initial_mean =
       std::vector<double>(6, 0.0);
    
    boost::shared_ptr<pcl::tracking::KLDAdaptiveParticleFilterOMPTracker<
       PointT, ParticleT> > tracker
       (new pcl::tracking::KLDAdaptiveParticleFilterOMPTracker<
        PointT, ParticleT>(4));

    ParticleT bin_size;
    bin_size.x = 0.1f;
    bin_size.y = 0.1f;
    bin_size.z = 0.1f;
    bin_size.roll = 0.1f;
    bin_size.pitch = 0.1f;
    bin_size.yaw = 0.1f;

    tracker->setMaximumParticleNum(200);
    tracker->setDelta(0.99);
    tracker->setEpsilon(0.2);
    tracker->setBinSize(bin_size);

    this->tracker_ = tracker;
    this->tracker_->setTrans(Eigen::Affine3f::Identity());
    this->tracker_->setStepNoiseCovariance(default_step_covariance);
    this->tracker_->setInitialNoiseCovariance(initial_noise_covariance);
    this->tracker_->setInitialNoiseMean(default_initial_mean);
    this->tracker_->setIterationNum(1);
    this->tracker_->setParticleNum(200);
    this->tracker_->setResampleLikelihoodThr(0.00);
    this->tracker_->setUseNormal(false);

    pcl::tracking::ApproxNearestPairPointCloudCoherence<PointT>::Ptr coherence =
       pcl::tracking::ApproxNearestPairPointCloudCoherence<PointT>::Ptr
       (new pcl::tracking::ApproxNearestPairPointCloudCoherence<PointT>());
    
    boost::shared_ptr<pcl::tracking::DistanceCoherence<PointT> > dist_coherence
       = boost::shared_ptr<pcl::tracking::DistanceCoherence<PointT> >(
          new pcl::tracking::DistanceCoherence<PointT>());
    coherence->addPointCoherence(dist_coherence);
    boost::shared_ptr<pcl::tracking::HSVColorCoherence<PointT> >
       hsv_color_coherence = boost::shared_ptr<
          pcl::tracking::HSVColorCoherence<PointT> >(
             new pcl::tracking::HSVColorCoherence<PointT>());
    coherence->addPointCoherence(hsv_color_coherence);
    boost::shared_ptr<pcl::search::Octree<PointT> > search(
       new pcl::search::Octree<PointT> (0.01));
    coherence->setSearchMethod(search);
    coherence->setMaximumDistance(0.01);
    this->tracker_->setCloudCoherence(coherence);

    this->choi_dataset_.clear();
    const char * cd_path = "/home/krishneel/Downloads/seq_synth_orange_juice_kitchen/data.txt";
    std::ifstream infile(cd_path);
    std::string line;
    while (std::getline(infile, line)) {
       std::istringstream iss(line);
       std::string path_to_pcd;
       iss >> path_to_pcd;
       this->choi_dataset_.push_back(path_to_pcd);
    }
    choi_counter_ = 0;
    
    
    this->onInit();
}

void ParticleFilters::onInit() {
    this->subscribe();
    this->pub_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>(
       "/choi/output/cloud", 1);
    this->pub_pose_ = pnh_.advertise<geometry_msgs::PoseStamped>(
       "/particle_filter_tracker/track_result_pose", 1);
}

void ParticleFilters::subscribe() {
    this->sub_cloud_ = this->pnh_.subscribe(
       "input_cloud", 1, &ParticleFilters::cloudCB, this);
   
    this->sub_templ_.subscribe(this->pnh_, "input_template", 1);
    this->sub_pose_.subscribe(this->pnh_, "input_pose", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(this->sub_templ_, this->sub_pose_);
    this->sync_->registerCallback(
       boost::bind(&ParticleFilters::templateCB, this, _1, _2));
}

void ParticleFilters::unsubscribe() {
   
}

void ParticleFilters::cloudCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    if (choi_dataset_.empty()) {
       ROS_WARN_ONCE("NO CHOI DATA");
       return;
    }

    if (choi_counter_ > choi_dataset_.size()) {
       ROS_WARN("ALL PROCESSED");
       ros::shutdown();
    }

    ROS_INFO("PROCESSING: %d", choi_counter_);
    
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(
       new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::io::loadPCDFile<pcl::PointXYZRGBA>(
       choi_dataset_[choi_counter_++], *cloud);
    sensor_msgs::PointCloud2 *ros_cloud = new sensor_msgs::PointCloud2;
    pcl::toROSMsg(*cloud, *ros_cloud);
    ros_cloud->header = cloud_msg->header;
    this->pub_cloud_.publish(*ros_cloud);

    if (choi_counter_ == 1) {
       ROS_WARN("FIRST FRAME SLEEP");
       ros::Duration(10).sleep();
    } else {
       ros::Duration(0.3).sleep();
    }
    
    return;
    /**
     * RUNNING CHOI:
     1) LAUNCH this node
     2) launch point_cloud_image_creator
     3) handheld_object_registration POINTS:=/choi/output/cloud
     IMAGE:=/cloud_image/output/image
     4) object_annotation with same as above input
     5) tracking_sample (launch is modified here)
     */
    ///


    
    // pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    boost::mutex::scoped_lock lock(mtx_);
    pcl::PointCloud<PointT>::Ptr cloud_pass(new pcl::PointCloud<PointT>);
    this->filterPassThrough(cloud, *cloud_pass);

    pcl::PointCloud<PointT>::Ptr cloud_pass_downsampled_(
       new pcl::PointCloud<PointT>);
    this->gridSampleApprox(cloud_pass,
                           *cloud_pass_downsampled_,
                           downsampling_grid_size_);
    
    if (!cloud_pass_downsampled_->empty()) {
       this->tracker_->setInputCloud(cloud_pass_downsampled_);
       this->tracker_->compute();
       ParticleT result = tracker_->getResult();
       Eigen::Affine3f transformation = this->tracker_->toEigenMatrix(result);

       tf::Transform tfTransformation;
       tf::transformEigenToTF((Eigen::Affine3d)transformation,
                              tfTransformation);
       geometry_msgs::PoseStamped result_pose_stamped;
       tf::poseTFToMsg(tfTransformation, result_pose_stamped.pose);
       result_pose_stamped.header = cloud_msg->header;
       
       pcl::PointCloud<PointT>::Ptr result_cloud(new pcl::PointCloud<PointT>);
       pcl::transformPointCloud<PointT>(
          *(this->tracker_->getReferenceCloud()), *result_cloud,
          transformation);
       
       // sensor_msgs::PointCloud2 *ros_cloud = new sensor_msgs::PointCloud2;
       pcl::toROSMsg(*result_cloud, *ros_cloud);
       ros_cloud->header = cloud_msg->header;

       this->pub_pose_.publish(result_pose_stamped);
       this->pub_cloud_.publish(*ros_cloud);

       this->prev_transformation_ = transformation;
    }
}

void ParticleFilters::templateCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::PointCloud2::ConstPtr &pose_msg
   /*,
     const geometry_msgs::PoseStamped::ConstPtr &pose_msg*/
    ) {

    // ROS_WARN("SETTING UP TEMPLATE");
    pcl::PointCloud<PointT>::Ptr target_cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *target_cloud);

    Eigen::Vector4f centroid;
    Eigen::Affine3f trans = Eigen::Affine3f::Identity();

    if (!tracker_init_) {
       pcl::compute3DCentroid<PointT>(*target_cloud, centroid);
       trans.translation().matrix() = Eigen::Vector3f(centroid[0],
                                                      centroid[1],
                                                      centroid[2]);
    } else {
       trans = this->prev_transformation_;
    }
    pcl::PointCloud<PointT>::Ptr transed_ref(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud<PointT>(*target_cloud,
                                     *transed_ref, trans.inverse());
    pcl::PointCloud<PointT>::Ptr transed_ref_downsampled(
       new pcl::PointCloud<PointT>);
    this->gridSampleApprox(transed_ref, *transed_ref_downsampled,
                           downsampling_grid_size_);
    this->tracker_->setReferenceCloud(transed_ref_downsampled);
    this->tracker_->setTrans(trans);
    // tracker_->resetTracking();

    tracker_init_ = true;
    
}

void ParticleFilters::filterPassThrough(
    const CloudConstPtr &cloud, Cloud &result) {
    pcl::PassThrough<PointT> pass;
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 3.0);
    pass.setKeepOrganized(false);
    pass.setInputCloud(cloud);
    pass.filter(result);
}

void ParticleFilters::gridSampleApprox(
    const CloudConstPtr &cloud, Cloud &result, double leaf_size) {
    pcl::ApproximateVoxelGrid<PointT> grid;
    grid.setLeafSize(static_cast<float>(leaf_size),
                     static_cast<float>(leaf_size),
                     static_cast<float>(leaf_size));
    grid.setInputCloud(cloud);
    grid.filter(result);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "particle_filter_tracking");
    ParticleFilters pft;
    ros::spin();
    return 0;
}
