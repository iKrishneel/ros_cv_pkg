
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

#include <geometry_msgs/PolygonStamped.h>
#include <jsk_recognition_msgs/Rect.h>

#include <omp.h>
#include <ros_cmt_tracker/CMT.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/video/background_segm.hpp>
#include <boost/thread/mutex.hpp>

class ROSCMTTracker: public CMT {

 private:
    cv::Rect_<int> screen_rect_;
    int block_size_;
    bool tracker_init_;
    bool object_init_;

    int frame_counter_;
    int down_size_;
   
 protected:

    ros::NodeHandle pnh_;
    ros::Publisher pub_image_;
    ros::Publisher pub_rect_;
    ros::Subscriber sub_image_;
    ros::Subscriber sub_screen_pt_;
   
    void onInit();
    void subscribe();
    void unsubscribe();
   
 public:
    ROSCMTTracker();
    virtual void callback(
       const sensor_msgs::Image::ConstPtr &);
    virtual void screenPointCallback(
       const geometry_msgs::PolygonStamped &);
   
};

