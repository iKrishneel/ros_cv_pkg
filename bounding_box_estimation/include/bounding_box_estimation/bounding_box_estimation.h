// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo, Japan

#ifndef _BOUNDING_BOX_ESTIMATION_H
#define _BOUNDING_BOX_ESTIMATION_H

#include <ros/ros.h>
#include <ros/console.h>
#include <boost/thread/mutex.hpp>

#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/thread/thread.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>

#include <bounding_box_estimation/moment_of_inertia_estimation.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <geometry_msgs/PoseStamped.h>

class BoundingBoxEstimation {

private:
    typedef pcl::PointXYZRGB PointT;
    
    void onInit();
    void subscribe();
    
    geometry_msgs::PoseStamped pose_;
    
public:
    BoundingBoxEstimation();
    virtual void callback(
        const sensor_msgs::PointCloud2::ConstPtr &);
    virtual void orientation(
    const geometry_msgs::PoseStamped::ConstPtr &);
    
protected:
    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_cloud_;
    ros::Subscriber sub_pose_;
    ros::Publisher pub_cloud_;
    ros::Publisher pub_bbox_;
};


#endif  // _BOUNDING_BOX_ESTIMATION_H
