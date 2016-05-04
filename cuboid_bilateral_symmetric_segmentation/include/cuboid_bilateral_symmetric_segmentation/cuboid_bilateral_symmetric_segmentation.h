
#ifndef _CUBOID_BILATERAL_SYMMETRIC_SEGMENTATION_H_
#define _CUBOID_BILATERAL_SYMMETRIC_SEGMENTATION_H_


#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/correspondence.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/distances.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>

#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_utils/geo/polygon.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>
#include <omp.h>

#include <cuboid_bilateral_symmetric_segmentation/supervoxel_segmentation.h>
#include <cuboid_bilateral_symmetric_segmentation/moment_of_inertia_estimation.h>
#include <cuboid_bilateral_symmetric_segmentation/oriented_bounding_box.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <boost/thread/mutex.hpp>

// namespace jsk_msgs = jsk_recognition_msgs;

class CuboidBilateralSymmetricSegmentation:
    public SupervoxelSegmentation,
    public OrientedBoundingBox {
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    typedef pcl::SupervoxelClustering<PointT>::VoxelAdjacencyList AdjacencyList;
   
 private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
  
    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2,
       sensor_msgs::PointCloud2,
       jsk_msgs::PolygonArray,
       jsk_msgs::ModelCoefficientsArray> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_normal_;
    message_filters::Subscriber<jsk_msgs::PolygonArray> sub_planes_;
    message_filters::Subscriber<jsk_msgs::ModelCoefficientsArray> sub_coef_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;
   
    typedef std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> SupervoxelMap;
    typedef std::map<uint32_t, int> UInt32Map;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();

    ros::Publisher pub_cloud_;
    ros::Publisher pub_edge_;
    ros::Publisher pub_indices_;
    ros::Publisher pub_bbox_;
   
 public:
    CuboidBilateralSymmetricSegmentation();
    void cloudCB(const sensor_msgs::PointCloud2::ConstPtr &,
                 const sensor_msgs::PointCloud2::ConstPtr &,
		 const jsk_msgs::PolygonArrayConstPtr &,
		 const jsk_msgs::ModelCoefficientsArrayConstPtr &);
    void supervoxelDecomposition(SupervoxelMap &,
                                 pcl::PointCloud<NormalT>::Ptr,
                                 const pcl::PointCloud<PointT>::Ptr);
    float coplanarityCriteria(const Eigen::Vector4f, const Eigen::Vector4f,
                              const Eigen::Vector4f, const Eigen::Vector4f,
                              const float = 10, const float = 0.02f);
    void updateSupervoxelClusters(SupervoxelMap &,
                                 const uint32_t, const uint32_t);
    void supervoxel3DBoundingBox(jsk_msgs::BoundingBox &, const SupervoxelMap &,
				 const jsk_msgs::PolygonArrayConstPtr &,
				 const jsk_msgs::ModelCoefficientsArrayConstPtr &,
				 const int);

};


#endif  // _CUBOID_BILATERAL_SYMMETRIC_SEGMENTATION_H_
