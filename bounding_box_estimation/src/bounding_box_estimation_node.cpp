// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo, Japan

#include <bounding_box_estimation/bounding_box_estimation.h>

BoundingBoxEstimation::BoundingBoxEstimation() {
    this->onInit();
}

void BoundingBoxEstimation::onInit() {
    this->subscribe();
    this->pub_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>(
        "/bounding_box_estimation/output/cloud", 1);

    this->pub_bbox_ = pnh_.advertise<jsk_recognition_msgs::BoundingBoxArray>(
        "/bounding_box_estimation/output/bounding_boxes", 1);
}

void BoundingBoxEstimation::subscribe() {
    // this->sub_pose_ = this->pnh_.subscribe(
    //     "input_pose", 1, &BoundingBoxEstimation::orientation, this);
    this->sub_cloud_ = this->pnh_.subscribe(
        "input", 1, &BoundingBoxEstimation::callback, this);
}


void BoundingBoxEstimation::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    // boost::mutex::scoped_lock lock(this->lock_);
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty()) {
        ROS_ERROR("EMPTY INPUT CLOUD");
        return;
    }
    pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
    feature_extractor.setInputCloud(cloud);
    feature_extractor.compute();

    std::vector <float> moment_of_inertia;
    std::vector <float> eccentricity;
    PointT min_point_AABB;
    PointT max_point_AABB;
    PointT min_point_OBB;
    PointT max_point_OBB;
    PointT position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    float major_value, middle_value, minor_value;
    Eigen::Vector3f major_vector, middle_vector, minor_vector;
    Eigen::Vector3f mass_center;
    feature_extractor.getMomentOfInertia(moment_of_inertia);
    feature_extractor.getEccentricity(eccentricity);
    feature_extractor.getAABB(min_point_AABB, max_point_AABB);
    feature_extractor.getOBB(min_point_OBB, max_point_OBB,
                              position_OBB, rotational_matrix_OBB);
    feature_extractor.getEigenValues(major_value, middle_value, minor_value);
    feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
    feature_extractor.getMassCenter(mass_center);

    Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
    Eigen::Quaternionf quat(rotational_matrix_OBB);
    float width = max_point_OBB.x - min_point_OBB.x;
    float height = max_point_OBB.y - min_point_OBB.y;
    float depth = max_point_OBB.z - min_point_OBB.z;

    jsk_recognition_msgs::BoundingBox bounding_box;
    bounding_box.pose.position.x = position_OBB.x;
    bounding_box.pose.position.y = position_OBB.y;
    bounding_box.pose.position.z = position_OBB.z;
    bounding_box.pose.orientation.x = quat.x();
    bounding_box.pose.orientation.y = quat.y();
    bounding_box.pose.orientation.z = quat.z();
    bounding_box.pose.orientation.w = quat.w();
    bounding_box.dimensions.x = width;
    bounding_box.dimensions.y = height;
    bounding_box.dimensions.z = depth;
    bounding_box.header = cloud_msg->header;
    
    jsk_recognition_msgs::BoundingBoxArray bounding_boxes;
    bounding_boxes.boxes.push_back(bounding_box);
    bounding_boxes.header = cloud_msg->header;
    pub_bbox_.publish(bounding_boxes);

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    pub_cloud_.publish(ros_cloud);
}

void BoundingBoxEstimation::orientation(
    const geometry_msgs::PoseStamped::ConstPtr & pose_mgs) {
    ROS_INFO("POSE RECEIVED");
    geometry_msgs::PoseStamped pose_ = *pose_mgs;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "bounding_box_estimation");
    BoundingBoxEstimation bbe;
    ros::spin();
}

