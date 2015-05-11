// Copyright (C) 2015 by Krishneel Chaudhary @ JSK

#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>
#include <pcl/common/impl/common.hpp>
#include <pcl/registration/distances.h>

#include <tf/transform_broadcaster.h>

#include <iostream>
#include <fstream>

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
#define CLEAR "\033[2J"  // clear screen escape code


typedef pcl::PointXYZRGB PointT;
/**
 * Function to convert Numbers to string to string
 */
template<typename T>
inline std::string convertNumber2String(T c_frame) {
    std::string frame_num;
    std::stringstream out;
    out << c_frame;
    frame_num = out.str();
    return frame_num;
}


/**
 *  Return a RGB colour value given a scalar v in the range [vmin,vmax]
 *  In this case each colour component ranges from 0 (no contribution) to
 *  1 (fully saturated), modifications for other ranges is trivial.
 *  The colour is clipped at the end of the scales if v is outside
 *  the range [vmin,vmax]
*/
template<typename T, typename U, typename V>
inline cv::Scalar JetColour(T v, U vmin, V vmax) {
    cv::Scalar c = cv::Scalar(1.0, 1.0, 1.0);  // white
    T dv;

    if (v < vmin)
       v = vmin;
    if (v > vmax)
       v = vmax;
    dv = vmax - vmin;

    if (v < (vmin + 0.25 * dv)) {
       c.val[0] = 0;
       c.val[1] = 4 * (v - vmin) / dv;
    } else if (v < (vmin + 0.5 * dv)) {
       c.val[0] = 0;
       c.val[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    } else if (v < (vmin + 0.75 * dv)) {
       c.val[0] = 4 * (v - vmin - 0.5 * dv) / dv;
       c.val[2] = 0;
    } else {
       c.val[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
       c.val[2] = 0;
    }
    return(c);
}

#endif   // _CONSTANTS_H_"

