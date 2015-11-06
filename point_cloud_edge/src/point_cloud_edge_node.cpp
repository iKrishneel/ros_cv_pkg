
#include <point_cloud_edge/point_cloud_edge.h>

PointCloudEdge::PointCloudEdge() {
   
    this->onInit();
    this->subscribe();
}

void PointCloudEdge::onInit() {
    this->pub_hc_edge_ = pnh_.advertise<sensor_msgs::PointCloud2>(
        "/point_cloud_edge/output/high_curvature_edge", 1);
    this->pub_oc_edge_ = pnh_.advertise<sensor_msgs::PointCloud2>(
       "/point_cloud_edge/output/occluding_edge", 1);

    this->pub_occluding_indices_ = pnh_.advertise<pcl_msgs::PointIndices>(
       "/point_cloud_edge/output/occluding_edge_indices", 1);
    this->pub_curvature_indices_ = pnh_.advertise<pcl_msgs::PointIndices>(
       "/point_cloud_edge/output/curvature_edge_indices", 1);
    
}

void PointCloudEdge::subscribe() {
    this->sub_cloud_ = this->pnh_.subscribe(
       "input_cloud", 1, &PointCloudEdge::callback, this);
}

void PointCloudEdge::callback(
    const sensor_msgs::PointCloud2::ConstPtr & cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    pcl::PointCloud<pcl::Normal>::Ptr normals(
       new pcl::PointCloud<pcl::Normal>);
    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
    ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
    ne.setNormalSmoothingSize(10.0f);
    ne.setBorderPolicy(ne.BORDER_POLICY_MIRROR);
    ne.setInputCloud(cloud);
    ne.compute(*normals);
    
    pcl::OrganizedEdgeFromRGBNormals<PointT, pcl::Normal, pcl::Label> oed;
    oed.setInputNormals(normals);
    oed.setInputCloud(cloud);
    oed.setDepthDisconThreshold(0.05);
    oed.setMaxSearchNeighbors(100);
    pcl::PointCloud<pcl::Label> labels;
    std::vector<pcl::PointIndices> label_indices;
    oed.compute(labels, label_indices);

    this->publishIndices(pub_oc_edge_, pub_occluding_indices_,
                         cloud, label_indices[1].indices, cloud_msg->header);
    this->publishIndices(pub_hc_edge_, pub_curvature_indices_,
                         cloud, label_indices[3].indices, cloud_msg->header);
}

void PointCloudEdge::publishIndices(
    ros::Publisher& pub, ros::Publisher& pub_indices,
    const pcl::PointCloud<PointT>::Ptr& cloud,
    const std::vector<int>& indices, const std_msgs::Header& header) {
    pcl_msgs::PointIndices msg;
    msg.header = header;
    msg.indices = indices;
    pub_indices.publish(msg);
    pcl::PointCloud<PointT>::Ptr output(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*cloud, indices, *output);
    sensor_msgs::PointCloud2 ros_output;
    pcl::toROSMsg(*output, ros_output);
    ros_output.header = header;
    pub.publish(ros_output);
  }

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "point_cloud_edge");
    PointCloudEdge pce;
    ros::spin();
}
