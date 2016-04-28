
#ifndef _GPU_PARTICLE_FILTER_H_
#define _GPU_PARTICLE_FILTER_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <geometry_msgs/PolygonStamped.h>
#include <jsk_recognition_msgs/Rect.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <particle_filter_tracking/particle_filter.h>
#include <particle_filter_tracking/histogram_of_oriented_gradients.h>

#include <omp.h>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/thread/mutex.hpp>

class ParticleFilterTracking: public ParticleFilter {

    struct PFFeatures {
        cv::Mat color_hist;
        cv::Mat hog_hist;
    };
    
 private:

    cv::Mat reference_histogram_;
    
    cv::Rect_<int> screen_rect_;
    bool tracker_init_;
    int width_;
    int height_;
    int block_size_;
    int downsize_;

    cv::Mat dynamics;
    std::vector<Particle> particles_;
    cv::RNG random_num_;
    cv::Mat prev_frame_;

    boost::shared_ptr<HOGFeatureDescriptor> hog_;
    PFFeatures reference_features_;
    
 protected:
    virtual void onInit();
    virtual void subscribe();
    virtual void unsubscribe();

    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_image_;
    ros::Subscriber sub_screen_pt_;
    ros::Publisher pub_image_;
    unsigned int threads_;
   
 public:
    ParticleFilterTracking();
    virtual void imageCB(
        const sensor_msgs::Image::ConstPtr &);
    virtual void screenPtCB(
        const geometry_msgs::PolygonStamped &);

    void initializeTracker(const cv::Mat &, cv::Rect &);
    void runObjectTracker(cv::Mat *image, cv::Rect &rect);
    void roiCondition(cv::Rect &, cv::Size);
    bool createParticlesFeature(PFFeatures &, const cv::Mat &,
                                const std::vector<Particle> &);
    void getHistogram(cv::Mat &, const cv::Mat &,
                      const int, const int, bool = true);
    std::vector<double> featureHistogramLikelihood(
       const std::vector<Particle> &, cv::Mat &,
       const PFFeatures, const PFFeatures);
    template<typename T>
    T EuclideanDistance(Particle, Particle, bool = true);
};

#endif  // _GPU_PARTICLE_FILTER_H_
