// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University of Tokyo

#ifndef _SCENE_DECOMPOSER_IMAGE_PROCESSOR_H_
#define _SCENE_DECOMPOSER_IMAGE_PROCESSOR_H_

#include <point_cloud_scene_decomposer/constants.h>
#include <point_cloud_scene_decomposer/connected.h>
#include <point_cloud_scene_decomposer/contour_thinning.h>

#include <vector>

template<typename T>
struct cvPatch {
    T k;  // number of clusters
    cv::Mat patch;  // pixel wise label
    cv::Rect_<T> rect;  // patch info
    bool is_region;  // mark region
    std::vector<cv::Point2i> region;
};

class SceneDecomposerImageProcessor {

 private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Subscriber depth_sub_;
    image_transport::Publisher image_pub_;
    image_transport::Publisher depth_pub_;
   
    int total_cluster;
    bool isPub;
    cv::RNG rng;
    cv::Size cell_size;

 protected:
    cv_bridge::CvImagePtr cvImgPtr;
    cv_bridge::CvImagePtr cvDepPtr;
    ConnectedComponents *connectedComponents;

    void onInit();
    void subscribe();
    void unsubscribe();
    
 public:
    SceneDecomposerImageProcessor();
    ~SceneDecomposerImageProcessor();
    virtual void imageCallback(
       const sensor_msgs::ImageConstPtr &);
    virtual void depthCallback(
       const sensor_msgs::ImageConstPtr &);

    virtual void getDepthEdge(
       const cv::Mat &, cv::Mat &, bool = true);
    virtual void cvMorphologicalOperations(
      const cv::Mat &, cv::Mat &, bool = true);
    virtual void getRGBEdge(
       const cv::Mat &,  cv::Mat &, std::string = "cvCanny");
    virtual void cvGetLabelImagePatch(
       const pcl::PointCloud<PointT>::Ptr,
       const cv::Mat &,
       const cv::Mat &,
       std::vector<cvPatch<int> > &);
    virtual int cvLabelImagePatch(
       const cv::Mat &,
       cv::Mat &);
    virtual void cvLabelEgdeMap(
       const pcl::PointCloud<PointT>::Ptr,
       const cv::Mat &,
       cv::Mat,
       std::vector<cvPatch<int> > &);
    virtual void edgeBoundaryAssignment(
      const pcl::PointCloud<PointT>::Ptr,
      const cv::Mat &,
      cv::Mat &,
      const cv::Rect_<int>);
   
    void cvVisualization(
       std::vector<cvPatch<int> > &,
       const cv::Size = cv::Size(640, 480),
       const std::string = "Label Map");

    int getTotalClusterSize();
    void publishROSImage(
       cv::Mat &, cv::Mat &);
   
    cv::Mat image;
    cv::Mat depth;
    cv::Scalar color[10000];
};
#endif  // _SCENE_DECOMPOSER_IMAGE_PROCESSOR_H_
