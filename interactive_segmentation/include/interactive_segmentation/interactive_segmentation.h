
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
#include <boost/graph/grid_graph.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/iteration_macros.hpp>

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
#include <pcl/octree/octree.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/filter.h>
#include <pcl/common/centroid.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/surface/mls.h>
#include <pcl/point_types_conversion.h>
#include <pcl/registration/distances.h>
#include <pcl/features/don.h>
#include <pcl/segmentation/segment_differences.h>
#include <pcl/segmentation/min_cut_segmentation.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <point_cloud_scene_decomposer/signal.h>

#include <std_msgs/Header.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int64.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PointStamped.h>

#include <interactive_segmentation/supervoxel_segmentation.h>
#include <interactive_segmentation/graph_cut_segmentation.h>
#include <interactive_segmentation/saliency_map_generator.h>

#include <omp.h>

class InteractiveSegmentation: public SupervoxelSegmentation,
                               public GraphCutSegmentation {

    typedef pcl::PointXYZRGB PointT;
    typedef  pcl::FPFHSignature33 FPFHS;
  
    struct EdgeParam {
       cv::Point index;
       float orientation;
       int contour_position;
       bool flag;
    };
   
    struct EdgeNormalDirectionPoint {
       cv::Point2f normal_pt1;
       cv::Point2f normal_pt2;
       cv::Point2f tangent_pt1;
       cv::Point2f tangent_pt2;
       EdgeNormalDirectionPoint(
          cv::Point2f np1 = cv::Point(),
          cv::Point2f np2 = cv::Point(),
          cv::Point2f t1 = cv::Point(),
          cv::Point2f t2 = cv::Point()) :
          normal_pt1(np1), normal_pt2(np2),
          tangent_pt1(t1), tangent_pt2(t2) {}
    };
   
 private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
    typedef pcl::SupervoxelClustering<PointT>::VoxelAdjacencyList AdjacencyList;
    typedef message_filters::sync_policies::ApproximateTime<
       sensor_msgs::Image,
      sensor_msgs::PointCloud2,
       sensor_msgs::PointCloud2> SyncPolicy;

    message_filters::Subscriber<sensor_msgs::Image> sub_image_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_normal_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;
   
    ros::Publisher pub_cloud_;
    ros::Publisher pub_prob_;
    ros::Publisher pub_voxels_;
    ros::Publisher pub_indices_;
    ros::Publisher pub_image_;
    ros::Publisher pub_pt_map_;

    ros::Subscriber sub_screen_pt_;
    cv::Point2i screen_pt_;
    bool is_init_;
  
    int min_cluster_size_;
    int num_threads_;
  
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
   
 public:
    InteractiveSegmentation();

   virtual void screenPointCallback(
      const geometry_msgs::PointStamped &);
  
    virtual void callback(
       const sensor_msgs::Image::ConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &);

    void computePointFPFH(
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::Normal>::Ptr,
       cv::Mat &) const;

    void surfelLevelObjectHypothesis(
        pcl::PointCloud<PointT>::Ptr,
        pcl::PointCloud<pcl::Normal>::Ptr,
        std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > &);

    void updateSupervoxelClusters(
       std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> &,
       const uint32_t, const uint32_t);

   
    bool attentionSurfelRegionPointCloudMask(
       const pcl::PointCloud<PointT>::Ptr, const Eigen::Vector4f,
       const std_msgs::Header, pcl::PointCloud<PointT>::Ptr,
       pcl::PointIndices::Ptr);
    void attentionSurfelRegionMask(
       const cv::Mat, const cv::Point2i, cv::Mat &, cv::Rect &);
    void objectMinCutSegmentation(
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<PointT>::Ptr);
    void surfelSamplePointWeightMap(
        const pcl::PointCloud<PointT>::Ptr,
        const pcl::PointCloud<pcl::Normal>::Ptr,
        const pcl::PointXYZRGBA &,
        const Eigen::Vector4f, cv::Mat &);
    void organizedMinCutMaxFlowSegmentation(
       pcl::PointCloud<PointT>::Ptr, const int);
  
    void computePointCloudCovarianceMatrix(
       const pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<PointT>::Ptr);


    void pointLevelSimilarity(
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::Normal>::Ptr, const std_msgs::Header);

    float whiteNoiseKernel(const float);

    void generateFeatureSaliencyMap(
        const cv::Mat &, cv::Mat &);
  
    void viewPointSurfaceNormalOrientation(
        pcl::PointCloud<PointT>::Ptr,
        const pcl::PointCloud<pcl::Normal>::Ptr,
        const Eigen::Vector3f, const Eigen::Vector3f);
    void normalNeigbourOrientation(
        const pcl::PointCloud<PointT>::Ptr,
        const pcl::PointCloud<pcl::Normal>::Ptr,
        pcl::PointCloud<PointT>::Ptr, const int);
   
    virtual Eigen::Vector4f cloudMeanNormal(
    const pcl::PointCloud<pcl::Normal>::Ptr,
    bool = false);
    bool localVoxelConvexityCriteria(
        Eigen::Vector4f, Eigen::Vector4f,
        Eigen::Vector4f, Eigen::Vector4f,
        const float = 0.0f);
    
    virtual void pointCloudEdge(
       pcl::PointCloud<PointT>::Ptr,
       const cv::Mat &, const int = 50);
    void computeEdgeCurvature(
       const cv::Mat &,
       const std::vector<std::vector<cv::Point> > &contours,
       std::vector<std::vector<cv::Point> > &,
       std::vector<std::vector<EdgeNormalDirectionPoint> >&);
    template<class T>
    void estimatePointCloudNormals(
       const pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<pcl::Normal>::Ptr,
       T = 0.05f, bool = false) const;

    void pointIntensitySimilarity(
       pcl::PointCloud<PointT>::Ptr,
       const int);
   
    void mlsSmoothPointCloud(
        const pcl::PointCloud<PointT>::Ptr,
        pcl::PointCloud<PointT>::Ptr,
        pcl::PointCloud<pcl::Normal>::Ptr);
};


#endif  // _INTERACTIVE_SEGMENTATION_H_
