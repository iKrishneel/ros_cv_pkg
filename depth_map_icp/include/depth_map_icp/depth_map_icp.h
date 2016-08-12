
#ifndef _DEPTH_MAP_ICP_H_
#define _DEPTH_MAP_ICP_H_

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <depth_map_icp/ICPOdometry.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/thread/mutex.hpp>

class DepthMapICP {

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

 private:
    boost::mutex mutex_;
    boost::mutex lock_;

    typedef  message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2, sensor_msgs::Image> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::Image> sub_depth_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_; 
   
    bool is_init_;
    cv::Mat1w prev_depth_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();

    ros::NodeHandle pnh_;
    ros::Publisher pub_cloud_;
   
 public:
    DepthMapICP();
    void depthCB(const sensor_msgs::PointCloud2::ConstPtr &,
                 const sensor_msgs::Image::ConstPtr &);
};


#endif  // _DEPTH_MAP_ICP_H_
