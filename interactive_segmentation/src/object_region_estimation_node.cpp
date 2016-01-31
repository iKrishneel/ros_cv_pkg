// Copyright (C) 2016 by Krishneel Chaudhary, JSK Lab,a
// The University of Tokyo, Japan

#include <interactive_segmentation/object_region_estimation_node.h>

ObjectRegionEstimation::ObjectRegionEstimation {
   
}

void ObjectRegionEstimation::onInit() {
   
}

void ObjectRegionEstimation::subscribe() {
   
}

void ObjectRegionEstimation::unsubscribe() {
   
}

void ObjectRegionEstimation::callback(
    const sensor_msgs::PointCloud2::ConstPtr &,
    const sensor_msgs::PointCloud2::ConstPtr &) {
   
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "object_region_estimation");
    ObjectRegionEstimation ore;
    ros::spin();
    return 0;
}
