
#ifndef _PARTICLE_FILTER_TRACKING_H_
#define _PARTICLE_FILTER_TRACKING_H_

#include <particle_filter_tracking/particle_filter.h>

#include <ros/ros.h>
#include <ros/console.h>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/videostab/videostab.hpp>
#include <opencv2/videostab/global_motion.hpp>
#include <opencv2/videostab/optical_flow.hpp>
#include <opencv2/video/video.hpp>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <boost/thread/mutex.hpp>

class ParticleFilterTracking: public ParticleFilter {

 private:
    virtual void onInit();
    virtual void subscribe();
    virtual void unsubscribe();

    
 public:
    ParticleFilterTracking();
    virtual void imageCallback(
       const sensor_msgs::Image::ConstPtr &);
   
 protected:
    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_image_;
    ros::Publisher pub_image_;
};

#endif  // _PARTICLE_FILTER_TRACKING_H_
