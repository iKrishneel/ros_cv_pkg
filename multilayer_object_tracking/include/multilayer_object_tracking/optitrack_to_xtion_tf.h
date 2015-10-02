// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo, Japan

#ifndef _OPTITRACK_TO_XTION_TF_H_
#define _OPTITRACK_TO_XTION_TF_H_

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

class OptiTrackToXtionTF {
 private:

    ros::NodeHandle pnh_;
    ros::Publisher pub_pose_;
    ros::Subscriber sub_pose_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
   
 public:
    OptiTrackToXtionTF();
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


#endif  //_OPTITRACK_TO_XTION_TF_H_
