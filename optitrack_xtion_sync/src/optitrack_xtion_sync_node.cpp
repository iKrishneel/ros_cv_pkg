// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo, Japan

#include <optitrack_xtion_sync/optitrack_xtion_sync.h>
#include <string>

OptiTrackXtionSync::OptiTrackXtionSync() {
    this->onInit();
}

void OptiTrackXtionSync::onInit() {
    this->subscribe();
}

void OptiTrackXtionSync::subscribe() {

    this->sub_pose_ = this->pnh_.subscribe(
       "/test/pose", 1, &OptiTrackXtionSync::callback, this);

    dynamic_reconfigure::Server<
       optitrack_xtion_sync::OptiTrackXtionSyncConfig>::CallbackType f =
       boost::bind(&OptiTrackXtionSync::configCallback, this, _1, _2);
    server.setCallback(f);
}

void OptiTrackXtionSync::unsubscribe() {
   
}

void OptiTrackXtionSync::callback(
    const geometry_msgs::PoseStampedConstPtr &pose_msg) {
    this->adjustmentXtionWorldToXtion();

    
    std::string xtion_tf = "/xtion/base_link";
    std::string object_tf = "/test/base_link";
    std::string world_tf = "/world";
    /*
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

void OptiTrackXtionSync::configCallback(
    optitrack_xtion_sync::OptiTrackXtionSyncConfig  &config, uint32_t level) {
    this->quaternion_x_ = static_cast<double>(config.quaternion_x);
    this->quaternion_y_ = static_cast<double>(config.quaternion_y);
    this->quaternion_z_ = static_cast<double>(config.quaternion_z);
    this->quaternion_w_ = static_cast<double>(config.quaternion_w);

    this->translation_x_ = static_cast<double>(config.translation_x);
    this->translation_y_ = static_cast<double>(config.translation_y);
    this->translation_z_ = static_cast<double>(config.translation_z);
}


void OptiTrackXtionSync::getXtionToObjectWorld(
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

void OptiTrackXtionSync::sendNewTFFrame(
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

void OptiTrackXtionSync::adjustmentXtionWorldToXtion() {
    static tf::TransformBroadcaster br;
    tf::Transform update_transform;
    tf::Vector3 origin = tf::Vector3(translation_x_,
                                     translation_y_,
                                     translation_z_);
    update_transform.setOrigin(origin);
    tf::Quaternion quat = tf::Quaternion(
       quaternion_x_, quaternion_y_, quaternion_z_, quaternion_w_);
    update_transform.setRotation(quat);
    br.sendTransform(tf::StampedTransform(
                        update_transform, ros::Time::now(),
                        "/xtion/base_link", "/camera_link"));

    std::cout << quaternion_x_ << ", " << quaternion_y_ << ", "
              << quaternion_z_ << std::endl;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "optitrack_to_xtion_tf");
    OptiTrackXtionSync otx;
    ros::spin();
    return 0;
}
