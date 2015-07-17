
#ifndef _INTERACTIVE_SEGMENTATION_H_
#define _INTERACTIVE_SEGMENTATION_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/video/background_segm.hpp>

#include <boost/thread/mutex.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/segment_differences.h>
#include <pcl/octree/octree.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/filter.h>
#include <pcl/common/centroid.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <point_cloud_scene_decomposer/signal.h>

#include <std_msgs/Bool.h>
#include <std_msgs/Int64.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>

#include <vector>
#include <string>

class InteractiveSegmentation {

    struct EdgeParam {
       cv::Point index;
       float orientation;
       int contour_position;
       bool flag;
    };
   
    struct EdgeNormalDirectionPoint {
       cv::Point pt;
       cv::Point end_pt;

       EdgeNormalDirectionPoint(
          cv::Point p = cv::Point(),
          cv::Point e = cv::Point()) :
          pt(p), end_pt(e) {}
    };
       
   
 private:
    typedef pcl::PointXYZRGB PointT;
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::Image,
       sensor_msgs::Image,
       sensor_msgs::PointCloud2> SyncPolicy;

    message_filters::Subscriber<sensor_msgs::Image> sub_image_;
    message_filters::Subscriber<sensor_msgs::Image> sub_edge_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    ros::Publisher pub_cloud_;
    ros::Publisher pub_image_;

 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
   
 public:
    InteractiveSegmentation();
    virtual void callback(
       const sensor_msgs::Image::ConstPtr &,
       const sensor_msgs::Image::ConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &);
    virtual void pointCloudEdge(
       const cv::Mat &, const cv::Mat &, const int = 50);
    virtual void cvMorphologicalOperations(
       const cv::Mat &, cv::Mat &, bool, int);

    void computeEdgeCurvature(
       const cv::Mat &,
       const std::vector<std::vector<cv::Point> > &contours,
       std::vector<std::vector<cv::Point> > &,
       std::vector<EdgeNormalDirectionPoint> &);
    void computeEdgeCurvatureOrientation(
       const std::vector<std::vector<cv::Point> > &,
       const std::vector<std::vector<cv::Point> > &,
       std::vector<std::vector<float> > &,
       bool = true);
    void getEdgeNormalPoint(
        cv::Mat &,
        std::vector<EdgeNormalDirectionPoint> &,
        const std::vector<std::vector<cv::Point> > &,
        const std::vector<std::vector<cv::Point> > &,
        const std::vector<std::vector<float> > &,
        const float = 10);
};


#endif  // _INTERACTIVE_SEGMENTATION_H_
