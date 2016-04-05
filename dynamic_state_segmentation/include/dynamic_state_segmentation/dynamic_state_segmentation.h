
#ifndef _DYNAMIC_STATE_SEGMENTATION_H_
#define _DYNAMIC_STATE_SEGMENTATION_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <boost/thread/mutex.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/distances.h>

#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

class DynamicStateSegmentation {

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
  
private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
  
    typedef  message_filters::sync_policies::ApproximateTime<
        sensor_msgs::PointCloud2,
        geometry_msgs::PointStamped> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<geometry_msgs::PointStamped> screen_pt_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    PointT seed_point_;
    NormalT seed_normal_;
    int seed_index_;

    int num_threads_;
    int neigbor_size_;
  
    pcl::KdTreeFLANN<PointT>::Ptr kdtree_;
  
protected:
    void onInit();
    void subscribe();
    void unsubscribe();

    ros::Publisher pub_cloud_;
  
public:
    DynamicStateSegmentation();
    virtual void cloudCB(const sensor_msgs::PointCloud2::ConstPtr &,
                         const geometry_msgs::PointStamped::ConstPtr &);
    void seedCorrespondingRegion(std::vector<int> &,
                                 const pcl::PointCloud<PointT>::Ptr,
                                 const pcl::PointCloud<NormalT>::Ptr, const int);
    void getPointNeigbour(std::vector<int> &, const pcl::PointCloud<PointT>::Ptr,
                          const PointT, const int = 8);
    int localVoxelConvexityCriteria(Eigen::Vector4f, Eigen::Vector4f,
                                    Eigen::Vector4f, Eigen::Vector4f,
                                    const float = 0.0f, bool = true);
    template<class T>
    void estimateNormals(const pcl::PointCloud<PointT>::Ptr,
                         pcl::PointCloud<NormalT>::Ptr,
                         T = 0.05f, bool = false) const;  

    /**
     * functions for CRF
     */
    void computeFeatures(cv::Mat &, const pcl::PointCloud<PointT>::Ptr,
                         const pcl::PointCloud<NormalT>::Ptr, const int);
    void pointColorContrast(const pcl::PointCloud<PointT>::Ptr,
                            const pcl::PointCloud<PointT>::Ptr, const int);

};
#endif  // _DYNAMIC_STATE_SEGMENTATION_H_
