
#ifndef _TEST_ICP_H_
#define _TEST_ICP_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <sensor_msgs/PointCloud2.h>
#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d_omp.h>

class TestICP {

protected:

void onInit();
void subscribe();
void unsubscribe();


private:

ros::NodeHandle nh_;
ros::Publisher pub_cloud_;
ros::Subscriber sub_cloud_;

public:
TestICP();
void cloudCallback(
const sensor_msgs::PointCloud2::ConstPtr &);


};


#endif  // _TEST_ICP_H_
