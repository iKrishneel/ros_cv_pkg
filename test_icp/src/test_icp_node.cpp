
#include <test_icp/test_icp.h>

TestICP::TestICP() {

}

void TestICP::onInit() {
  this->pub_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "/test_icp/output/cloud", sizeof(char));
}

void TestICP::subscribe() {
  this->sub_cloud_ = nh_.subscribe("input", 1,
                                   &TestICP::cloudCallback, this);
  
}

void TestICP::unsubscribe() {

}

void TestICP::cloudCallback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msgs) {
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

int main(int argc, char *argv[]) {

  ros::init(argc, argv, "test_icp");
  TestICP ticp;
  ros::spin();
  return 0;
}
