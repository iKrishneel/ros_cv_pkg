
#include <object_recognition/point_cloud_object_detection.h>
#include <object_recognition/object_recognition.h>
#include <jsk_recognition_msgs/Rect.h>
#include <object_recognition/NonMaximumSuppression.h>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <algorithm>
#include <utility>

ObjectRecognition::ObjectRecognition() :
    it_(nh_),
    supportVectorMachine_(new cv::SVM) {

    nh_.getParam("trainer_manifest", this->trainer_manifest_filename_);
    this->readTrainingManifestFromDirectory();
    
    this->nms_client_ = nh_.serviceClient<
       object_recognition::NonMaximumSuppression>("non_maximum_suppression");
    
    nh_.getParam("run_type", this->run_type_);
    // nh_.getParam("/object_recognition/run_type", this->run_type_);
    
    try {
       ROS_INFO("%s--Loading Trained SVM Classifier%s", GREEN, RESET);
       // this->model_name_ = "dataset/drill_hog_lbp_svm.xml";
       this->supportVectorMachine_->load(this->model_name_.c_str());
       ROS_INFO("%s--Classifier Loaded Successfully%s", GREEN, RESET);
    } catch(cv::Exception &e) {
       ROS_ERROR("%s--ERROR: %s%s", BOLDRED, e.what(), RESET);
       std::_Exit(EXIT_FAILURE);
    }
    this->subscribe();

    this->pub_rects_ = this->nh_.advertise<
       jsk_recognition_msgs::RectArray>(
          "/object_detection/output/rects", 1);
    this->image_pub_ = this->it_.advertise(
       "/image_annotation/output/segmentation", 1);
}

void ObjectRecognition::subscribe() {
    this->image_sub_ = this->it_.subscribe(
       "input", 1, &ObjectRecognition::imageCb, this);
    
    dynamic_reconfigure::Server<
       object_recognition::ObjectDetectionConfig>::CallbackType f =
       boost::bind(&ObjectRecognition::configCallback, this, _1, _2);
    server.setCallback(f);
}

void ObjectRecognition::imageCb(
    const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return;
    }
    cv::Mat image;
    cv::Size isize = cv_ptr->image.size();
    // control params
    const int downsize = 2;
    const float scale = this->scale_;
    const int img_stack = this->stack_size_;
    const int incrementor = this->incrementor_;

    cv::resize(cv_ptr->image, image, cv::Size(
                  isize.width/downsize, isize.height/downsize));
    std::multimap<float, cv::Rect_<int> > detection_info =
       this->runObjectRecognizer(image, cv::Size(
                                    this->swindow_x, this->swindow_y),
                                 scale, img_stack, incrementor);
    cv::Mat dimg = image.clone();
    ROS_INFO("--Info Size: %ld", detection_info.size());
    for (std::multimap<float, cv::Rect_<int> >::iterator
            it = detection_info.begin(); it != detection_info.end(); it++) {
       cv::rectangle(dimg, it->second, cv::Scalar(0, 255, 0), 2);
    }
    if (this->run_type_.compare("DETECTOR") == 0) {
       const float nms_threshold = 0.01;
       std::vector<cv::Rect_<int> > object_rects = this->nonMaximumSuppression(
          detection_info, nms_threshold);
       cv::Mat bimg = image.clone();
       for (std::vector<cv::Rect_<int> >::iterator it = object_rects.begin();
            it != object_rects.end(); it++) {
          cv::rectangle(bimg, *it, cv::Scalar(0, 0, 255), 2);
       }
       jsk_recognition_msgs::RectArray jsk_rect_array;
       this->convertCvRectToJSKRectArray(
          object_rects, jsk_rect_array, downsize, isize);
       jsk_rect_array.header = msg->header;
       this->pub_rects_.publish(jsk_rect_array);
       cv::imshow("Final Detection", bimg);
    } else if (this->run_type_.compare("BOOTSTRAPER") == 0) {
       this->bootstrapFalsePositiveDetection(
          image, this->ndataset_path_, this->nonobject_dataset_filename_,
          detection_info);
    } else {
       ROS_ERROR("NODELET RUNTYPE IS NOT SET");
       std::_Exit(EXIT_FAILURE);
    }
    cv::imshow("Initial Detection", dimg);
    cv::waitKey(3);
    
    this->image_pub_.publish(cv_ptr->toImageMsg());
}

std::multimap<float, cv::Rect_<int> > ObjectRecognition::runObjectRecognizer(
    const cv::Mat &image, const cv::Size wsize,
    const float scale, const int scale_counter, const int incrementor) {
    if (image.empty()) {
       ROS_ERROR("%s--INPUT IMAGE IS EMPTY%s", RED, RESET);
       return std::multimap<float, cv::Rect_<int> >();
    }
    cv::Size nwsize = wsize;
    int scounter = 0;
    std::multimap<float, cv::Rect_<int> > detection_info;
    int sw_incrementor = incrementor;
    while (scounter++ < scale_counter) {
       this->objectRecognizer(image, detection_info, nwsize, sw_incrementor);
       this->pyramidialScaling(nwsize, scale);
       sw_incrementor += (sw_incrementor * scale);
    }
    return detection_info;
}

void ObjectRecognition::objectRecognizer(
    const cv::Mat &image, std::multimap<float, cv::Rect_<int> > &detection_info,
    const cv::Size wsize, const int incrementor) {
    for (int j = 0; j < image.rows; j += incrementor) {
       for (int i = 0; i < image.cols; i += incrementor) {
          cv::Rect_<int> rect = cv::Rect_<int>(i, j, wsize.width, wsize.height);
          if ((rect.x + rect.width <= image.cols) &&
              (rect.y + rect.height <= image.rows)) {
             cv::Mat roi = image(rect).clone();
             cv::GaussianBlur(roi, roi, cv::Size(3, 3), 1.0);
             cv::resize(roi, roi, cv::Size(this->swindow_x, this->swindow_y));
             cv::Mat hog_feature = this->computeHOG(roi);
             cv::Mat hsv_feature;
             this->computeHSHistogram(roi, hsv_feature, 16, 16, true);
             hsv_feature = hsv_feature.reshape(1, 1);
             cv::Mat _feature = hog_feature;
             this->concatenateCVMat(hog_feature, hsv_feature, _feature);
             float response = this->supportVectorMachine_->predict(
                _feature, false);
             if (response == 1) {
                detection_info.insert(std::make_pair(response, rect));
             } else {
                continue;
             }
          }
       }
    }
}


/**
 * color histogram temp placed here
 */
void ObjectRecognition::computeHSHistogram(
    cv::Mat &src, cv::Mat &hist, const int hBin, const int sBin, bool is_norm) {
    if (src.empty()) {
       return;
    }
    cv::Mat hsv;
    cv::cvtColor(src, hsv, CV_BGR2HSV);
    int histSize[] = {hBin, sBin};
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    const float* ranges[] = {h_ranges, s_ranges};
    int channels[] = {0, 1};
    cv::calcHist(
       &hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    if (is_norm) {
       cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    }
}

void ObjectRecognition::pyramidialScaling(
    cv::Size &wsize, const float scale) {
    float nwidth = wsize.width + (wsize.width * scale);
    float nheight = wsize.height + (wsize.height * scale);
    const int min_swindow_size = 16;
    nwidth = (nwidth < min_swindow_size) ? min_swindow_size : nwidth;
    nheight = (nheight < min_swindow_size) ? min_swindow_size : nheight;
    wsize = cv::Size(std::abs(nwidth), std::abs(nheight));
}

std::vector<cv::Rect_<int> > ObjectRecognition::nonMaximumSuppression(
    std::multimap<float, cv::Rect_<int> > &detection_info,
    const float nms_threshold) {
    if (detection_info.empty()) {
       return std::vector<cv::Rect_<int> >();
    }
    object_recognition::NonMaximumSuppression srv_nms;
    std::vector<jsk_recognition_msgs::Rect> rect_msg;
    for (std::multimap<float, cv::Rect_<int> >::iterator
            it = detection_info.begin(); it != detection_info.end(); it++) {
       cv::Rect_<int> cv_rect = it->second;
       jsk_recognition_msgs::Rect jsk_rect;
       jsk_rect.x = cv_rect.x;
       jsk_rect.y = cv_rect.y;
       jsk_rect.width = cv_rect.width;
       jsk_rect.height = cv_rect.height;
       srv_nms.request.rect.push_back(jsk_rect);
       srv_nms.request.probabilities.push_back(it->first);
    }
    srv_nms.request.threshold = nms_threshold;;
    std::vector<cv::Rect_<int> > bbox;
    if (this->nms_client_.call(srv_nms)) {
       for (int i = 0; i < srv_nms.response.bbox_count; i++) {
          cv::Rect_<int> brect = cv::Rect_<int>(
             srv_nms.response.bbox[i].x,
             srv_nms.response.bbox[i].y,
             srv_nms.response.bbox[i].width,
             srv_nms.response.bbox[i].height);
          bbox.push_back(brect);
       }
    } else {
       ROS_ERROR("Failed to call service add_two_ints");
       return std::vector<cv::Rect_<int> >();
    }
    return bbox;
}

void ObjectRecognition::concatenateCVMat(
    const cv::Mat &mat_1, const cv::Mat &mat_2,
    cv::Mat &featureMD, bool iscolwise) {
    if (iscolwise) {
       featureMD = cv::Mat(mat_1.rows, (mat_1.cols + mat_2.cols), CV_32F);
        for (int i = 0; i < featureMD.rows; i++) {
           for (int j = 0; j < mat_1.cols; j++) {
              featureMD.at<float>(i, j) = mat_1.at<float>(i, j);
           }
           for (int j = mat_1.cols; j < featureMD.cols; j++) {
              featureMD.at<float>(i, j) = mat_2.at<float>(i, j - mat_1.cols);
           }
        }
    } else {
       featureMD = cv::Mat((mat_1.rows + mat_2.rows), mat_1.cols, CV_32F);
       for (int i = 0; i < featureMD.cols; i++) {
          for (int j = 0; j < mat_1.rows; j++) {
             featureMD.at<float>(j, i) = mat_1.at<float>(j, i);
          }
          for (int j = mat_1.rows; j < featureMD.rows; j++) {
             featureMD.at<float>(j, i) = mat_2.at<float>(j - mat_1.rows, i);
          }
       }
    }
}

void ObjectRecognition::objectBoundingBoxPointCloudIndices(
    const std::vector<cv::Rect_<int> > &bounding_boxes,
    std::vector<pcl::PointIndices> &cluster_indices,
    const int downsize,
    const cv::Size img_sz) {
    cluster_indices.clear();
    for (std::vector<cv::Rect_<int> >::const_iterator it =
            bounding_boxes.begin(); it != bounding_boxes.end(); it++) {
       int x = it->x * downsize;
       int y = it->y * downsize;
       int w = it->width * downsize;
       int h = it->height * downsize;
       pcl::PointIndices _indices;
       for (int j = y; j < (y + h); j++) {
          for (int i = x; i < (x + w); i++) {
             int _index = (i + (j * img_sz.width));
             _indices.indices.push_back(_index);
          }
       }
       cluster_indices.push_back(_indices);
       _indices.indices.clear();
    }
}

void ObjectRecognition::convertCvRectToJSKRectArray(
      const std::vector<cv::Rect_<int> > &bounding_boxes,
      jsk_recognition_msgs::RectArray &jsk_rects,
      const int downsize, const cv::Size img_sz) {
      for (std::vector<cv::Rect_<int> >::const_iterator it =
              bounding_boxes.begin(); it != bounding_boxes.end(); it++) {
         jsk_recognition_msgs::Rect j_r;
         j_r.x = it->x * downsize;
         j_r.y = it->y * downsize;
         j_r.width = it->width * downsize;
         j_r.height = it->height * downsize;
         jsk_rects.rects.push_back(j_r);
      }
}

void ObjectRecognition::configCallback(
    object_recognition::ObjectDetectionConfig &config, uint32_t level) {
    boost::mutex::scoped_lock lock(mutex_);
    this->scale_ = static_cast<float>(config.scaling_factor);
    this->stack_size_ = static_cast<int>(config.stack_size);
    this->incrementor_ = config.sliding_window_increment;

    // this->run_type_ = config.run_type;
    
    // this->model_name_ = config.svm_model_name;
    
      // currently fixed variables
    // this->swindow_x = config.sliding_window_width;
    // this->swindow_y = config.sliding_window_height;
    this->downsize_ = config.image_downsize;
}

void ObjectRecognition::bootstrapFalsePositiveDetection(
    const cv::Mat &frame, std::string save_path, std::string rw_filename,
    const std::multimap<float, cv::Rect_<int> > &detection_info) {
    if (frame.empty()) {
       ROS_ERROR("-- CANNOT BOOTSTRAP EMPTY DATA");
       return;
    }
    int icounter = 0;
    std::ofstream b_outFile(rw_filename.c_str(), std::ios_base::app);
    for (std::multimap<float, cv::Rect_<int> >::const_iterator
            it = detection_info.begin(); it != detection_info.end(); it++) {
       cv::Mat roi = frame(it->second).clone();
       /* -- chnage name to data-time wise */
       std::string sve_pt ="img_" + this->convertNumber2String<int>(icounter++)
          + ".jpg";
       cv::imwrite(save_path + sve_pt, roi);
       b_outFile << sve_pt << std::endl;
    }
    b_outFile.close();
}

template<typename T>
std::string ObjectRecognition::convertNumber2String(
    T c_frame) {
    std::string frame_num;
    std::stringstream out;
    out << c_frame;
    frame_num = out.str();
    return frame_num;
}

void ObjectRecognition::readTrainingManifestFromDirectory() {
    cv::FileStorage fs = cv::FileStorage(
       this->trainer_manifest_filename_, cv::FileStorage::READ);
    if (!fs.isOpened()) {
       ROS_ERROR("TRAINER MANIFEST NOT FOUND..");
       std::_Exit(EXIT_FAILURE);
    }
    cv::FileNode n = fs["TrainerInfo"];
    std::string ttype = n["trainer_type"];
    std::string tpath = n["trainer_path"];
    this->model_name_ = tpath;  // classifier path
    
    n = fs["FeatureInfo"];  // features used
    int hog = static_cast<int>(n["HOG"]);
    int lbp = static_cast<int>(n["LBP"]);

    n = fs["SlidingWindowInfo"];  // window size
    int sw_x = static_cast<int>(n["swindow_x"]);
    int sw_y = static_cast<int>(n["swindow_y"]);
    this->swindow_x = sw_x;
    this->swindow_y = sw_y;
    
    n = fs["TrainingDatasetDirectoryInfo"];
    std::string pfile = n["object_dataset_filename"];
    std::string nfile = n["nonobject_dataset_filename"];
    std::string dataset_path = n["dataset_path"];
    this->object_dataset_filename_ = pfile;  // obj/non dataset
    this->nonobject_dataset_filename_ = nfile;
    this->ndataset_path_ = dataset_path;  // name to database/negative    
}

int main(int argc, char *argv[]) {
   
    ros::init(argc, argv, "object_recognition");
    ROS_INFO("%sRUNNING OBJECT RECOGNITION NODELET%s", BOLDRED, RESET);
    ObjectRecognition recognition;
    PointCloudObjectDetection pcod;
    ros::spin();
    return 0;
}
