// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo, Japan

#include <multilayer_object_tracking/optitrack_to_xtion_tf.h>
#include <string>

OptiTrackToXtionTF::OptiTrackToXtionTF() {
    this->onInit();
}

void OptiTrackToXtionTF::onInit() {
    this->subscribe();
}

void OptiTrackToXtionTF::subscribe() {

    this->sub_pose_ = this->pnh_.subscribe(
       "/test/pose", 1, &OptiTrackToXtionTF::callback, this);
   
}

void OptiTrackToXtionTF::unsubscribe() {
   
}

void OptiTrackToXtionTF::callback(
    const geometry_msgs::PoseStampedConstPtr &pose_msg) {
    this->adjustmentXtionWorldToXtion();

    
    std::string xtion_tf = "/xtion/base_link";
    std::string object_tf = "/test/base_link";
    std::string world_tf = "/world";
    ros::Time now = ros::Time(0);
    tf::StampedTransform transform;
    this->getXtionToObjectWorld("/camera_link", xtion_tf, now, transform);
    tf::Quaternion tf_quaternion =  transform.getRotation();
    tf::Vector3 origin = tf::Vector3(transform.getOrigin().getX(),
                                     transform.getOrigin().getY(),
                                     transform.getOrigin().getZ());
    tf::Quaternion quat = tf::Quaternion(
       tf_quaternion.x(), tf_quaternion.y(),
       tf_quaternion.z(), tf_quaternion.w());

    std::cout << tf_quaternion.x() << ", " << tf_quaternion.y() << ", "
              << tf_quaternion.z() << ", " <<  tf_quaternion.w() << std::endl;
    /*
    // ------
    tf::StampedTransform world_object_transform;
    this->getXtionToObjectWorld(
       world_tf, object_tf, now, world_object_transform);
    tf::Quaternion wot_quaternion =  world_object_transform.getRotation();
    tf::Vector3 wot_origin = tf::Vector3(
       world_object_transform.getOrigin().getX(),
       world_object_transform.getOrigin().getY(),
       world_object_transform.getOrigin().getZ());
    tf::Quaternion wot_quat = tf::Quaternion(
       wot_quaternion.x(), wot_quaternion.y(),
       wot_quaternion.z(), wot_quaternion.w());

    
    // this->sendNewTFFrame(origin, quat, pose_msg->header.stamp,
    //                      "/xtion_world_frame", "/world_frame");
    // this->sendNewTFFrame(wot_origin, wot_quat, pose_msg->header.stamp,
    //                      "/world_frame", "/xtion_object_frame", true);
    ros::Time tim = ros::Time::now();
    this->sendNewTFFrame(origin, quat, tim,
                         "/xtion_world_frame", "/world_frame");
                         
    this->sendNewTFFrame(wot_origin, wot_quat, tim,
                         "/world_frame", "/xtion_object_frame", true);
    */
}


void OptiTrackToXtionTF::getXtionToObjectWorld(
    const std::string parent, const std::string child,
    const ros::Time now, tf::StampedTransform &transform) {
    tf::TransformListener tf_listener;
    bool wft_ok = tf_listener.waitForTransform(
        child, parent, now, ros::Duration(2.0f));
    if (!wft_ok) {
        ROS_ERROR("CANNOT TRANSFORM SOURCE AND TARGET FRAMES");
        return;
    }
    tf_listener.lookupTransform(
       child, parent, now, transform);
    
}

void OptiTrackToXtionTF::sendNewTFFrame(
    const tf::Vector3 trans, const tf::Quaternion quaternion,
    const ros::Time now, std::string parent, std::string new_frame,
    bool is_inverse) {
    tf::Transform update_transform;
    tf::Vector3 origin = trans;
    update_transform.setOrigin(origin);
    tf::Quaternion quat = quaternion;
    update_transform.setRotation(quat);
    static tf::TransformBroadcaster br;
    if (is_inverse) {
       br.sendTransform(tf::StampedTransform(
                           update_transform.inverse(), now,
                           parent, new_frame));
    } else {
       br.sendTransform(tf::StampedTransform(
                           update_transform, now,
                           parent, new_frame));
    }
}

void OptiTrackToXtionTF::adjustmentXtionWorldToXtion() {
    static tf::TransformBroadcaster br;
    tf::Transform update_transform;
    tf::Vector3 origin = tf::Vector3(0.01, 0.01, -0.01);
    update_transform.setOrigin(origin);
    tf::Quaternion quat = tf::Quaternion(
       0.0, 0.0, 0.0, 1.0);
    update_transform.setRotation(quat);
    br.sendTransform(tf::StampedTransform(
                        update_transform, ros::Time::now(),
                        "/xtion/base_link", "/camera_link"));
    
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "optitrack_to_xtion_tf");
    OptiTrackToXtionTF otx;
    ros::spin();
    return 0;
}
