
#ifndef _TEMP_ROBOT_NODE_H_
#define _TEMP_ROBOT_NODE_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <opencv2/opencv.hpp>

#include <boost/thread/mutex.hpp>

#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

#include <point_cloud_scene_decomposer/signal.h>

class RobotNode {
    
 public:
    
    RobotNode();

    virtual void onInit();
    virtual void subsribe();

    virtual void signalCallback(
       const point_cloud_scene_decomposer::signal &);
    
 protected:
    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_cloud_;
    ros::Publisher pub_cloud_;
    
 private:
    int processing_counter_;
};

#endif  // _TEMP_ROBOT_NODE_H_
