
#ifndef _COLLISION_CHECK_GRASP_PLANNAR_H_
#define _COLLISION_CHECK_GRASP_PLANNAR_H_

#include <omp.h>
#include <ros/ros.h>
#include <ros/console.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/distances.h>
#include <pcl/common/centroid.h>
#include <pcl/registration/icp.h>

#include <std_msgs/Header.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>

namespace jsk_msgs = jsk_recognition_msgs;

class CollisionCheckGraspPlannar {

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    typedef pcl::PointXYZRGBNormal PointNormalT;
    typedef pcl::PointCloud<PointT> PointCloud;

    struct IndicesMap {
       int label;
       int index;
    };

    struct Facets {
       Eigen::Vector3f AA;
       Eigen::Vector3f AB;
       Eigen::Vector3f BB;
       Eigen::Vector3f BA;
    };

    struct SortVector {
      bool operator() (int i, int j) {
         return (i < j);
      }
    } sortVector;

#define NUMBER_OF_SIDE 4
   
 private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2, jsk_msgs::ClusterPointIndices,
       jsk_msgs::BoundingBoxArray> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<jsk_msgs::BoundingBoxArray> sub_boxes_;
    message_filters::Subscriber<jsk_msgs::ClusterPointIndices> sub_indices_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    pcl::KdTreeFLANN<PointT>::Ptr kdtree_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();

    ros::Publisher pub_cloud_;
    ros::Publisher pub_edge_;
    ros::Publisher pub_indices_;
    ros::Publisher pub_bbox_;
   
 public:
    CollisionCheckGraspPlannar();
    void cloudCB(const sensor_msgs::PointCloud2::ConstPtr &,
                 const jsk_msgs::ClusterPointIndices::ConstPtr &,
                 const jsk_msgs::BoundingBoxArray::ConstPtr &);
    void getBoundingBoxGraspPoints(std::vector<Eigen::Vector3f> &,
                                   const jsk_msgs::BoundingBox);
    PointT vector3f2PointT(const Eigen::Vector3f,
                           Eigen::Vector3f = Eigen::Vector3f(0, 0, 0));
    template<class T>
    void getPointNeigbour(std::vector<int> &, const PointT,
                          const T = 8, bool = true);
   
};


#endif  // _COLLISION_CHECK_GRASP_PLANNAR_H_

