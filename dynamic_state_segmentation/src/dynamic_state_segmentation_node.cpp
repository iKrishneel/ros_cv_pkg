
#include <dynamic_state_segmentation/dynamic_state_segmentation.h>

DynamicStateSegmentation::DynamicStateSegmentation() {

  printf("HELLO\n");
}

void DynamicStateSegmentation::onInit() {
  
}

void DynamicStateSegmentation::subscribe() {
  
}

void DynamicStateSegmentation::unsubscribe() {
  
}

void DynamicStateSegmentation::cloudCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
  
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, " dynamic_state_segmentation");
  DynamicStateSegmentation dss;
  ros::spin();

  printf("HELLO\n");
  return 0;
}


