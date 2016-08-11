
#ifndef _DEPTH_MAP_ICP_H_
#define _DEPTH_MAP_ICP_H_

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/thread/mutex.hpp>

class DepthMapICP {

 private:
    boost::mutex mutex_;
    boost::mutex lock_;

    bool is_init_;
    cv::Mat prev_depth_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();

    ros::NodeHandle pnh_;
    ros::Publisher pub_depth_;
    ros::Subscriber sub_depth_;
   
   
 public:
    DepthMapICP();
    void depthCB(const sensor_msgs::Image::ConstPtr &);
};


#endif  // _DEPTH_MAP_ICP_H_
