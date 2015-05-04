
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

ObjectRecognition::ObjectRecognition(const std::string dataset_path) :
    it_(nh_),
    dataset_path_(dataset_path),
    supportVectorMachine_(new cv::SVM) {

    this->nms_client_ = nh_.serviceClient<
       object_recognition::NonMaximumSuppression>("non_maximum_suppression");    
    
    bool isTrain = false;
    if (isTrain) {
       ROS_INFO("%s--Training Classifier%s", GREEN, RESET);
       trainObjectClassifier();
       ROS_INFO("%s--Trained Successfully..%s", BLUE, RESET);
       exit(-1);
    } else {
       ROS_INFO("%s--Loading Trained SVM Classifier%s", GREEN, RESET);
       try{
           this->model_name_ = "svm.xml";
           this->supportVectorMachine_->load((this->model_name_).c_str());
           ROS_INFO("%s--Trained SVM Classifier Loaded Successfully%s", GREEN, RESET);
       } catch(cv::Exception &e){
           ROS_ERROR("%s--ERROR LOADING TRAINED CLASSIFIER %s%s", BOLDRED, e.what(), RESET);
           std::_Exit(EXIT_FAILURE);
       }
    }
    this->subscribe();

    this->pub_rects_ = this->nh_.advertise<
       jsk_recognition_msgs::RectArray>(
          "/object_detection/output/rects", 1);
    /*
    this->pub_indices_ = this->nh_.advertise<
       jsk_recognition_msgs::ClusterPointIndices>(
          "/object_detection/indices/output", 1);
    */
    this->image_pub_ = this->it_.advertise(
       "/image_annotation/output/segmentation", 1);
}

void ObjectRecognition::subscribe() {
    this->image_sub_ = this->it_.subscribe(
       "/camera/rgb/image_rect_color"
       /*"/multisense/left/image_rect_color"*/, 1,
       &ObjectRecognition::imageCb, this);

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
    const int downsize = this->downsize_;
    const float scale = this->scale_;
    const int img_stack = this->stack_size_;
    const int incrementor = this->incrementor_;

    cv::resize(cv_ptr->image, image, cv::Size(
                  isize.width/downsize, isize.height/downsize));
    std::vector<cv::Rect_<int> > bb_rects = this->runObjectRecognizer(
       image, cv::Size(this->swindow_x, this->swindow_y),
       scale, img_stack, incrementor);

    jsk_recognition_msgs::RectArray jsk_rect_array;
    this->convertCvRectToJSKRectArray(
       bb_rects, jsk_rect_array, downsize, isize);
    jsk_rect_array.header = msg->header;

    /*
    std::vector<pcl::PointIndices> bb_cluster_indices;
    this->objectBoundingBoxPointCloudIndices(
       bb_rects, bb_cluster_indices, downsize, isize);
    jsk_recognition_msgs::ClusterPointIndices ros_indices;
    ros_indices.cluster_indices = pcl_conversions::convertToROSPointIndices(
       bb_cluster_indices, msg->header);
    this->pub_indices_.publish(ros_indices);
    ros_indices.header = msg->header;
    */
    
    this->pub_rects_.publish(jsk_rect_array);
    this->image_pub_.publish(cv_ptr->toImageMsg());
}

std::vector<cv::Rect_<int> >  ObjectRecognition::runObjectRecognizer(
    const cv::Mat &image, const cv::Size wsize,
    const float scale, const int scale_counter, const int incrementor) {
    if (image.empty()) {
       ROS_ERROR("%s--INPUT IMAGE IS EMPTY%s", RED, RESET);
       return image;
    }
    cv::Size nwsize = wsize;
    int scounter = 0;
    std::multimap<float, cv::Rect_<int> > detection_info;
    int sw_incrementor = incrementor;
    while (scounter++ < scale_counter) {
       // std::cout << "New Incrementing Size: " << sw_incrementor << std::endl;
       this->objectRecognizer(image, detection_info, nwsize, sw_incrementor);
       this->pyramidialScaling(nwsize, scale);
       sw_incrementor += (sw_incrementor * scale);
    }

    cv::Mat dimg = image.clone();
    std::cout << "Info Size: " << detection_info.size()  << std::endl;
    for (std::multimap<float, cv::Rect_<int> >::iterator
            it = detection_info.begin(); it != detection_info.end(); it++) {
       cv::rectangle(dimg, it->second, cv::Scalar(0, 255, 0), 2);
    }
    const float nms_threshold = 0.05;
    std::vector<cv::Rect_<int> > object_rects = this->nonMaximumSuppression(
       detection_info, nms_threshold);
    cv::Mat bimg = image.clone();
    for (std::vector<cv::Rect_<int> >::iterator it = object_rects.begin();
         it != object_rects.end(); it++) {
       cv::rectangle(bimg, *it, cv::Scalar(0, 0, 255), 2);
    }
    cv::imshow("Initial Detection", dimg);
    cv::imshow("Final Detection", bimg);
    cv::waitKey(3);

    return object_rects;
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
             cv::Mat roi_hog = this->computeHOG(roi);
             cv::Mat roi_lbp = this->computeLBP(
                 roi, cv::Size(8, 8), 10, false, true);
             cv::Mat _feature;
             this->concatenateCVMat(roi_hog, roi_lbp, _feature);
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

void ObjectRecognition::trainObjectClassifier() {
    // reading the positive training image
    std::string pfilename = dataset_path_ + "dataset/train.txt";
    std::vector<cv::Mat> pdataset_img;
    cv::Mat labelMD;
    this->readDataset(pfilename, pdataset_img, labelMD, true, 1);
    
    // reading the negative training image
    std::string nfilename = dataset_path_ + "dataset/negative.txt";
    std::vector<cv::Mat> ndataset_img;
    this->readDataset(nfilename, ndataset_img, labelMD, true, -1);

    pdataset_img.insert(
       pdataset_img.end(), ndataset_img.begin(), ndataset_img.end());
       
    cv::Mat featureMD;
    this->extractFeatures(pdataset_img, featureMD);
    std::cout << featureMD.size() << std::endl;
        
    try {
       this->trainBinaryClassSVM(featureMD, labelMD);
       this->supportVectorMachine_->save(
          (this->model_name_).c_str());
    } catch(std::exception &e) {
       ROS_ERROR("%s--ERROR: %s%s", BOLDRED, e.what(), RESET);
    }
}

void ObjectRecognition::readDataset(
    std::string filename, std::vector<cv::Mat> &dataset_img, cv::Mat &labelMD,
    bool is_usr_label, const int usr_label) {
    ROS_INFO("%s--READING DATASET IMAGE%s", GREEN, RESET);
    std::ifstream infile;
    infile.open(filename.c_str(), std::ios::in);
    char buffer[255];
    if (!infile.eof()) {
       while (infile.good()) {
          infile.getline(buffer, 255);
          std::string _line(buffer);
          if (!_line.empty()) {
             std::istringstream iss(_line);
             std::string _path;
             iss >> _path;
             cv::Mat img = cv::imread(this->dataset_path_+ _path,
                                      CV_LOAD_IMAGE_COLOR);
             float label;
             if (!is_usr_label) {
                std::string _label;
                iss >> _label;
                label = std::atoi(_label.c_str());
             } else {
                label = static_cast<float>(usr_label);
             }
             if (img.data) {
                labelMD.push_back(label);
                dataset_img.push_back(img);
             }
          }
       }
    }
}

/**
 * currently programmed using fixed sized image
 */
void ObjectRecognition::extractFeatures(
    const std::vector<cv::Mat> &dataset_img, cv::Mat &featureMD) {
    ROS_INFO("%s--EXTRACTING IMAGE FEATURES.%s", GREEN, RESET);
    for (std::vector<cv::Mat>::const_iterator it = dataset_img.begin();
         it != dataset_img.end(); it++) {
       cv::Mat img = *it;
       cv::resize(img, img, cv::Size(this->swindow_x, this->swindow_y));
       if (img.data) {
          cv::Mat hog_feature = this->computeHOG(img);
          cv::Mat lbp_feature = this->computeLBP(
             img, cv::Size(8, 8), 10, false, true);
          cv::Mat _feature;
          this->concatenateCVMat(hog_feature, lbp_feature, _feature, true);
          featureMD.push_back(_feature);
       }
       cv::imshow("image", img);
       cv::waitKey(3);
    }
}

void ObjectRecognition::trainBinaryClassSVM(
    const cv::Mat &featureMD, const cv::Mat &labelMD) {
    ROS_INFO("%s--TRAINING CLASSIFIER%s", GREEN, RESET);

    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;
    param.degree = 3;
    param.gamma = 0;
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-4;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 1;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
        
    svm_problem svm_prob_vector = this->convertCvMatToLibSVM(
        featureMD, labelMD, param);
    struct svm_model *model = new svm_model;
    if (svm_check_parameter(&svm_prob_vector, &param)) {
        std::cout << "ERROR" << std::endl;
    } else {
        model = svm_train(&svm_prob_vector, &param);
    }
    
    
    cv::SVMParams svm_param = cv::SVMParams();
    svm_param.svm_type = cv::SVM::NU_SVC;
    svm_param.kernel_type = cv::SVM::RBF;
    svm_param.degree = 0.0;
    svm_param.gamma = 0.90;
    svm_param.coef0 = 0.50;
    svm_param.C = 100;
    svm_param.nu = 0.70;
    svm_param.p = 1.0;
    svm_param.class_weights = NULL;
    svm_param.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
    svm_param.term_crit.max_iter = 1000;
    svm_param.term_crit.epsilon = 1e-6;
    cv::ParamGrid paramGrid = cv::ParamGrid();
    paramGrid.min_val = 0;
    paramGrid.max_val = 0;
    paramGrid.step = 1;

    /*this->supportVectorMachine_->train(
       featureMD, labelMD, cv::Mat(), cv::Mat(), svm_param);*/
    this->supportVectorMachine_->train_auto
       (featureMD, labelMD, cv::Mat(), cv::Mat(), svm_param, 10,
        paramGrid, cv::SVM::get_default_grid(cv::SVM::GAMMA),
        cv::SVM::get_default_grid(cv::SVM::P),
        cv::SVM::get_default_grid(cv::SVM::NU),
        cv::SVM::get_default_grid(cv::SVM::COEF),
        cv::SVM::get_default_grid(cv::SVM::DEGREE),
        true);
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
      const int downsize, const cv::Size img_sz)
   {
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

    this->model_name_ = config.svm_model_name;
    this->dataset_path_ = config.dataset_directory;
      
      // currently fixed variables
    this->swindow_x = config.sliding_window_width;
    this->swindow_y = config.sliding_window_height;
    this->downsize_ = config.image_downsize;
}


/**
 * Opencv Mat to libSVM svm_problem wrapper for training
 */
struct svm_problem ObjectRecognition::convertCvMatToLibSVM(
    const cv::Mat &feature_mat, const cv::Mat &label_mat,
    const svm_parameter &param) {
    if (feature_mat.rows != label_mat.rows) {
        std::cout << "--TRAINING SET IS NOT EQUIVALENT.." << std::endl;
        std::_Exit(EXIT_FAILURE);
    }
    svm_problem svm_prob_vector;
    const int feature_lenght = static_cast<int>(feature_mat.rows);
    const int feature_dimensions = static_cast<int>(feature_mat.cols);
    svm_prob_vector.l = feature_lenght;
    svm_prob_vector.y = new double[feature_lenght];
    svm_prob_vector.x = new svm_node*[feature_lenght];
    for (int i = 0; i < feature_lenght; i++) {
        svm_prob_vector.x[i] = new svm_node[feature_dimensions];
    }
    for (int j = 0; j < feature_lenght; j++) {
        svm_prob_vector.y[j] = static_cast<double>(label_mat.at<float>(j, 0));
        for (int i = 0; i < feature_dimensions; i++) {
            svm_prob_vector.x[j][i].index = i + 1;
            svm_prob_vector.x[j][i].value = static_cast<double>(
                feature_mat.at<float>(j, i));
        }
        svm_prob_vector.x[j][feature_dimensions].index = -1;
    }
    return svm_prob_vector;
}


int main(int argc, char *argv[]) {
   
    ros::init(argc, argv, "object_recognition");
    ROS_INFO("%sRUNNING OBJECT RECOGNITION NODELET%s", BOLDRED, RESET);
    std::string _path = "/home/krishneel/catkin_ws/src/image_annotation_tool/";
    ObjectRecognition recognition(_path);
    PointCloudObjectDetection pcod;
    ros::spin();
    return 0;
}
