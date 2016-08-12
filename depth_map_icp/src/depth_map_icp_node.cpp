
#include <depth_map_icp/depth_map_icp.h>

DepthMapICP::DepthMapICP() :
    is_init_(false) {
   
    this->onInit();
}

void DepthMapICP::onInit() {
    this->subscribe();
    this->pub_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>(
       "target", 1);
}

void DepthMapICP::subscribe() {
    // this->sub_depth_ = this->pnh_.subscribe(
    //    "depth", 1, &DepthMapICP::depthCB, this);

    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_depth_.subscribe(this->pnh_, "input_depth", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
                                        SyncPolicy> >(100);
    this->sync_->connectInput(this->sub_cloud_, this->sub_depth_);
    this->sync_->registerCallback(
       boost::bind(&DepthMapICP::depthCB, this, _1, _2));
}

void DepthMapICP::unsubscribe() {
    this->sub_depth_.unsubscribe();
    this->sub_cloud_.unsubscribe();
}

void DepthMapICP::depthCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::Image::ConstPtr &image_msg) {
   
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
   
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(
          image_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    const float MAX_DIST = 10.0f;
    const float MIN_DIST = 0.0f;
    cv::Mat d_image = cv::Mat::zeros(cv_ptr->image.size(), CV_8UC1);
    // cv_ptr->image.convertTo(d_image, CV_8UC1,
    //                         255/(MAX_DIST - MIN_DIST), MIN_DIST);


    PointCloud::Ptr region_cloud(new PointCloud);
    for (int j = 0; j < d_image.rows; j++) {
       for (int i = 0; i < d_image.cols; i++) {
          float dist = cv_ptr->image.at<float>(j, i);
          if (isnan(dist) || dist >  1.5f) {
             dist = MIN_DIST;
          } else {
             region_cloud->push_back(cloud->points[i + (j * d_image.cols)]);
          }

          // dist = (isnan(dist)) ? MIN_DIST : dist;
          // dist = (dist > 1.50) ? MIN_DIST : dist;
          d_image.at<uchar>(j, i) =  (dist / MAX_DIST) * 255;
       }
    }

    cv::Mat1w depth_image(d_image);
    // for (int j = 0; j < d_image.rows; j++) {
    //     for (int i = 0; i < d_image.cols; i++) {
    //        depth_image.at<unsigned short>(j, i) /= 5;
    //    }
    // }
    
    if (!is_init_) {
       ROS_WARN("SETTING THE INIT FRAME");
       this->prev_depth_ = depth_image.clone();
       is_init_ = true;
       return;
    }


    /*
    std::string path =
       "/home/krishneel/Downloads/icp-fast/rgbd_dataset_freiburg1_desk/depth/";
    
    cv::Mat1w prev_depth_1 = cv::imread(path + "1305031453.374112.png",
                             CV_LOAD_IMAGE_ANYCOLOR);
    cv::Mat1w depth_image1 = cv::imread(path + "1305031453.404816.png",
                             CV_LOAD_IMAGE_ANYCOLOR);
    */
    
    
    
    ICPOdometry icpOdom(640, 480, 320, 240, 528, 528);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int threads = 128;
    int blocks = 96;

    ROS_WARN("RUNNING ICP");
    
    Eigen::Matrix4f currPoseFast = Eigen::Matrix4f::Identity();
    icpOdom.initICPModel((unsigned short *)prev_depth_.data, MAX_DIST,
                         currPoseFast);
    icpOdom.initICP((unsigned short *)depth_image.data, MAX_DIST);
    
    Eigen::Vector3f trans = currPoseFast.topRightCorner(3, 1);
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot =
       currPoseFast.topLeftCorner(3, 3);
    
    icpOdom.getIncrementalTransformation(trans, rot, threads, blocks);
    
    currPoseFast.topLeftCorner(3, 3) = rot;
    currPoseFast.topRightCorner(3, 1) = trans;
    
    std::cout << currPoseFast  << "\n--------\n";
    
    // this->prev_depth_ = depth_image.clone();

    Eigen::Matrix4f inverse = currPoseFast.inverse();
    pcl::transformPointCloud(*region_cloud, *region_cloud, inverse);

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*region_cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
    
    cv::namedWindow("depth", cv::WINDOW_NORMAL);
    cv::imshow("depth", d_image);
    cv::waitKey(3);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "depth_map_icp");
    DepthMapICP dmi;
    ros::spin();
    return 0;
}


