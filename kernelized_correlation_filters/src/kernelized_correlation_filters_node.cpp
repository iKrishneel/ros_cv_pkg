
#include <kernelized_correlation_filters/kernelized_correlation_filters.h>


KernelizedCorrelationFilters::KernelizedCorrelationFilters() :
    block_size_(8), downsize_(1), tracker_init_(false), threads_(8) {
    this->tracker_ = boost::shared_ptr<KCF_Tracker>(new KCF_Tracker);

    std::string pretrained_weights;
    this->pnh_.getParam("pretrained_weights", pretrained_weights);
    if (pretrained_weights.empty()) {
       // ROS_FATAL("PROVIDE PRETRAINED WEIGHTS");
       // return;
    }
    std::string model_prototxt;
    this->pnh_.getParam("model_prototxt", model_prototxt);
    if (model_prototxt.empty()) {
       // ROS_FATAL("PROVIDE NETWORK PROTOTXT");
       // return;
    }
    std::string imagenet_mean;
    this->pnh_.getParam("imagenet_mean", imagenet_mean);
    if (imagenet_mean.empty()) {
       // ROS_ERROR("PROVIDE IMAGENET MEAN VALUE");
       // return;
    }
    int device_id;
    this->pnh_.param<int>("device_id", device_id, 0);
    this->pnh_.getParam("device_id", device_id);
    
    std::vector<std::string> feat_ext_layers(1);
    feat_ext_layers[0] = "conv1";

    /*
    //! test
    std::string caffe_root = "/home/krishneel/caffe/";
    pretrained_weights = caffe_root +
       "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
    model_prototxt = caffe_root +
       "models/bvlc_reference_caffenet/deploy.prototxt";
    imagenet_mean = caffe_root +
       "python/caffe/imagenet/imagenet_mean.binaryproto";

    ROS_WARN("SETTING CAFFE");
    FeatureExtractor *feature_extractor =
       new FeatureExtractor(pretrained_weights, model_prototxt,
                            imagenet_mean, feat_ext_layers, 0);
    ROS_WARN("CAFFE SETUP COMPLETED");

    cv::Mat image = cv::imread(caffe_root + "examples/images/cat.jpg");
    std::vector<cv::Mat> filters;
    feature_extractor->getFeatures(filters, image);

    std::cout << filters.size()  << "\n";
    
    //! end test
    */
    this->onInit();
}

void KernelizedCorrelationFilters::onInit() {
     this->subscribe();
     this->pub_image_ = pnh_.advertise<sensor_msgs::Image>(
         "target", 1);
}

void KernelizedCorrelationFilters::subscribe() {
     this->sub_screen_pt_ = this->pnh_.subscribe(
         "input_screen", 1, &KernelizedCorrelationFilters::screenPtCB, this);
     this->sub_image_ = this->pnh_.subscribe(
         "image", 1, &KernelizedCorrelationFilters::imageCB, this);
}

void KernelizedCorrelationFilters::unsubscribe() {
     this->sub_image_.shutdown();
}

void KernelizedCorrelationFilters::screenPtCB(
     const geometry_msgs::PolygonStamped &screen_msg) {
     int x = screen_msg.polygon.points[0].x;
     int y = screen_msg.polygon.points[0].y;
     int width = screen_msg.polygon.points[1].x - x;
     int height = screen_msg.polygon.points[1].y - y;
     this->screen_rect_ = cv::Rect_<int>(
         x/downsize_, y/downsize_, width/downsize_, height/downsize_);
     if (width > this->block_size_ && height > this->block_size_) {
         this->tracker_init_ = true;
     } else {
         ROS_WARN("-- Selected Object Size is too small... Not init tracker");
     }
}

void KernelizedCorrelationFilters::imageCB(
     const sensor_msgs::Image::ConstPtr &image_msg) {
     cv_bridge::CvImagePtr cv_ptr;
     try {
         cv_ptr = cv_bridge::toCvCopy(
             image_msg, sensor_msgs::image_encodings::BGR8);
     } catch (cv_bridge::Exception& e) {
         ROS_ERROR("cv_bridge exception: %s", e.what());
         return;
     }
     cv::Mat image = cv_ptr->image.clone();
     if (image.empty()) {
         ROS_ERROR("EMPTY INPUT IMAGE");
         return;
     }
     if (downsize_ > 1) {
         cv::resize(image, image, cv::Size(image.cols/this->downsize_,
                                           image.rows/this->downsize_));
     }

     if (this->tracker_init_) {
        ROS_INFO("Initializing Tracker");
        this->tracker_->init(image, this->screen_rect_);
        this->tracker_init_ = false;
        this->prev_frame_ = image(screen_rect_).clone();
        ROS_INFO("Tracker Initialization Complete");
     }

     if (this->screen_rect_.width > this->block_size_) {
        std::clock_t start;
        double duration;
        start = std::clock();
        
        this->tracker_->track(image);

        duration = (std::clock() - start) /
           static_cast<double>(CLOCKS_PER_SEC);
        ROS_INFO("PROCESS: %3.5f", duration);
        
        
        BBox_c bb = this->tracker_->getBBox();
        cv::Rect rect = cv::Rect(bb.cx - bb.w/2.0f,
                                 bb.cy - bb.h/2.0f, bb.w, bb.h);
        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
        
        
     } else {
        ROS_ERROR_ONCE("THE TRACKER IS NOT INITALIZED");
     }

     cv_bridge::CvImagePtr pub_msg(new cv_bridge::CvImage);
     pub_msg->header = image_msg->header;
     pub_msg->encoding = sensor_msgs::image_encodings::BGR8;
     pub_msg->image = image.clone();
     this->pub_image_.publish(pub_msg);

     cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
     cv::imshow("image", image);
     cv::waitKey(3);
}


int main(int argc, char *argv[]) {
   
    ros::init(argc, argv, "kernelized_correlation_filters");
    KernelizedCorrelationFilters kcf;
    ros::spin();
    return 0;
}

