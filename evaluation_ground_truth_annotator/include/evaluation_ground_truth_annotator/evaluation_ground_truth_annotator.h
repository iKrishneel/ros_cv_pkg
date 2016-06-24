
#ifndef _EVALUATION_GROUND_TRUTH_ANNOTATOR_H_
#define _EVALUATION_GROUND_TRUTH_ANNOTATOR_H_

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/distances.h>
#include <pcl/common/centroid.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>

#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Header.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_utils/geo/polygon.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>
#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <omp.h>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

namespace jsk_msgs = jsk_recognition_msgs;

class CvAlgorithmEvaluation {

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    typedef pcl::PointCloud<PointT> PointCloud;

 private:
    boost::mutex mutex_;
    boost::mutex lock_;
    ros::NodeHandle pnh_;
   
    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2, jsk_msgs::ClusterPointIndices> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<jsk_msgs::ClusterPointIndices> sub_indices_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    //! ground truth variables
    PointCloud::Ptr marked_cloud_;
    int labels_;
    std::string save_path_;
    std::vector<pcl::PointIndices> all_indices_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();

    ros::Publisher pub_cloud_;
    ros::Publisher pub_indices_;
    ros::Subscriber sub_gt_cloud_;
   
 public:
    CvAlgorithmEvaluation();
    void cloudCB(const sensor_msgs::PointCloud2::ConstPtr &,
                 const jsk_msgs::ClusterPointIndices::ConstPtr &);
    void intersectionUnionCoef(const PointCloud::Ptr, const PointCloud::Ptr,
                               const std::vector<pcl::PointIndices>,
                               const std::vector<pcl::PointIndices>);
    void groundTCB(const sensor_msgs::PointCloud2::ConstPtr &);
    void readBagFile(const std::string bag_path);
   
    template<class T>
    void getPointNeigbour(std::vector<int> &, const PointT,
                          const PointCloud::Ptr, const T = 8, bool = true);
   
};


#endif  // _EVALUATION_GROUND_TRUTH_ANNOTATOR
