// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo, Japan

#ifndef _OPTITRACK_XTION_SYNC_H_
#define _OPTITRACK_XTION_SYNC_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PoseStamped.h>

#include <dynamic_reconfigure/server.h>
#include <optitrack_xtion_sync/OptiTrackXtionSyncConfig.h>

class OptiTrackXtionSync {
 private:

    ros::NodeHandle pnh_;
    ros::Publisher pub_pose_;
    ros::Subscriber sub_pose_;

    double quaternion_x_;
    double quaternion_y_;
    double quaternion_z_;
    double quaternion_w_;

    double translation_x_;
    double translation_y_;
    double translation_z_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();

    dynamic_reconfigure::Server<
       optitrack_xtion_sync::OptiTrackXtionSyncConfig>  server;
    virtual void configCallback(
       optitrack_xtion_sync::OptiTrackXtionSyncConfig &, uint32_t);
   
 public:
    OptiTrackXtionSync();
    void callback(const geometry_msgs::PoseStampedConstPtr &);
    void getXtionToObjectWorld(
    const std::string , const std::string, const ros::Time,
       tf::StampedTransform &);
    void adjustmentXtionWorldToXtion();
    void sendNewTFFrame(
    const tf::Vector3 trans, const tf::Quaternion quat,
       const ros::Time now, std::string parent, std::string new_frame,
       bool = false);

};


#endif  //_OPTITRACK_XTION_SYNC_H_
