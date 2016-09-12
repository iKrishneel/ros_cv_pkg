
#ifndef _ANNOTATE_SYMMETRIC_PLANE_H_
#define _ANNOTATE_SYMMETRIC_PLANE_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

#include <geometry_msgs/PointStamped.h>

#include <opencv2/opencv.hpp>

class AnnotateSymmetricPlane {
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

 private:
    std::vector<cv::Point2i> selected_points2d_;
    bool is_init_;
   
 protected:
   
    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_cloud_;
    ros::Subscriber screen_pt_;
    ros::Publisher pub_cloud_;
   
    virtual void onInit();
    virtual void subscribe();
   
 public:
    AnnotateSymmetricPlane();
    void cloudCB(
       const sensor_msgs::PointCloud2::ConstPtr &);
    void screenCB(
      const geometry_msgs::PointStamped::ConstPtr &);
};


#endif /* _ANNOTATE_SYMMETRIC_PLANE_H_ */
