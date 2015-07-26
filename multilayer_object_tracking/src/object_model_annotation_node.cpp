
#include <multilayer_object_tracking/object_model_annotation.h>

ObjectModelAnnotation::ObjectModelAnnotation() {

    this->subscribe();
    this->onInit();
}

void ObjectModelAnnotation::onInit() {
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/object_model/output/cloud", 1);
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/object_model/output/image", 1);
    this->pub_pose_ = this->pnh_.advertise<geometry_msgs::PoseStamped>(
       "/object_model/output/pose", 1);
}

void ObjectModelAnnotation::subscribe() {
       this->sub_image_.subscribe(this->pnh_, "input_image", 1);
       this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
       this->sub_screen_pt_.subscribe(this->pnh_, "input_screen", 1);
       this->sync_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy> >(100);
       sync_->connectInput(sub_image_, sub_cloud_, sub_screen_pt_);
       sync_->registerCallback(boost::bind(&ObjectModelAnnotation::callback,
                                           this, _1, _2, _3));
}

void ObjectModelAnnotation::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_image_.unsubscribe();
}


void ObjectModelAnnotation::callback(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const geometry_msgs::PolygonStampedConstPtr &screen_msg) {
   // boost::mutex::scoped_lock lock(this->mutex_);
    int x = screen_msg->polygon.points[0].x;
    int y = screen_msg->polygon.points[0].y;
    int width = screen_msg->polygon.points[1].x - x;
    int height = screen_msg->polygon.points[1].y - y;
    cv::Rect_<int> screen_rect = cv::Rect_<int>(x, y, width, height);
    if (width < 24 && height < 24) {
       ROS_WARN("-- Selected Object Size is too small... Not init tracker");
       return;
    }
    cv::Mat image = cv_bridge::toCvShare(
       image_msg, image_msg->encoding)->image;
    
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    this->getAnnotatedObjectCloud(cloud, image, screen_rect);

    Eigen::Vector4f centroid;
    this->compute3DCentroids(cloud, centroid);
    geometry_msgs::PoseStamped ros_pose;
    ros_pose.pose.position.x = centroid(0);
    ros_pose.pose.position.y = centroid(1);
    ros_pose.pose.position.z = centroid(2);
    ros_pose.pose.orientation.x = 0.0f;
    ros_pose.pose.orientation.y = 0.0f;
    ros_pose.pose.orientation.z = 0.0f;
    ros_pose.pose.orientation.w = 0.0f;
    ros_pose.header = cloud_msg->header;
    
    cv_bridge::CvImage pub_img(
       image_msg->header, sensor_msgs::image_encodings::BGR8, image);
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;

    ROS_INFO("--Publish selected object info.");
    this->pub_cloud_.publish(ros_cloud);
    this->pub_image_.publish(pub_img.toImageMsg());
    this->pub_pose_.publish(ros_pose);
}

void ObjectModelAnnotation::getAnnotatedObjectCloud(
    pcl::PointCloud<PointT>::Ptr cloud,
    const cv::Mat &image,
    const cv::Rect_<int> screen_rect) {
    if (cloud->empty() || image.empty()) {
       ROS_ERROR("-- Cannot Process Empty Cloud");
       return;
    }
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    this->imageToPointCloudIndices(cloud, indices, image.size(), screen_rect);
    pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(*cloud, *tmp_cloud);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> euclidean_clustering;
    euclidean_clustering.setClusterTolerance(0.01);
    euclidean_clustering.setMinClusterSize(60);
    euclidean_clustering.setMaxClusterSize(25000);
    euclidean_clustering.setSearchMethod(tree);
    euclidean_clustering.setInputCloud(cloud);
    euclidean_clustering.setIndices(indices);
    euclidean_clustering.extract(cluster_indices);
    int max_size = FLT_MIN;
    int index = 0;
    for (int i = 0; i < cluster_indices.size(); i++) {
       int c_size = cluster_indices[i].indices.size();
       if (c_size > max_size) {
          max_size = c_size;
          index = i;
       }
    }
    cloud->clear();
    for (int i = 0; i < cluster_indices[index].indices.size(); i++) {
       int pi = cluster_indices[index].indices[i];
       cloud->points.push_back(tmp_cloud->points[pi]);
    }
}

void ObjectModelAnnotation::imageToPointCloudIndices(
    pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointIndices::Ptr indices,
    const cv::Size size, const cv::Rect_<int> rect) {
    if (cloud->empty()) {
       ROS_ERROR("-- Cannot Process Empty Cloud");
       return;
    }
    for (int j = rect.y; j < (rect.y + rect.height); j++) {
       for (int i = rect.x; i < (rect.x + rect.width); i++) {
          int index = i + (j * size.width);
          PointT pt = cloud->points[index];
          if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z)) {
             indices->indices.push_back(index);
          }
       }
    }
    for (int i = 0; i < cloud->size(); i++) {
       PointT pt = cloud->points[i];
       if (isnan(pt.x) || isnan(pt.y) || isnan(pt.z)) {
          pt.x = 0;
          pt.y = 0;
          pt.z = 0;
          cloud->points[i] = pt;
       }
    }
}

void ObjectModelAnnotation::compute3DCentroids(
    const pcl::PointCloud<PointT>::Ptr cloud,
    Eigen::Vector4f &centre) {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: empty cloud for centroid");
       centre = Eigen::Vector4f(-1, -1, -1, -1);
       return;
    }
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid<PointT, float>(*cloud, centroid);
    if (!isnan(centroid(0)) && !isnan(centroid(1)) && !isnan(centroid(2))) {
       centre = centroid;
    } else {
       ROS_ERROR("ERROR: NAN CENTROID\n-- reselect object");
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "object_model_annotation");
    ObjectModelAnnotation oma;
    ros::spin();
    return 0;
}
