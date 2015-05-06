
#include <image_annotation_tool/image_annotation_tool.h>

#define DEBUG

ImageAnnotationTool::ImageAnnotationTool(const std::string _path) :
    it_(nh_),
    iteration_counter_(0),
    model_save_counter_(0),
    mark_model_(true),
    segment_template_(true),
    write_txt_name_("train.txt"),
    save_folder_name_("dataset"),
    obj_rect_(cv::Rect_<int>(0, 0, 0, 0)) {

    bool is_dir_ok = this->saveDirectoryHandler(
       /*_path +*/ this->save_folder_name_, this->model_save_path_);
    std::cout << "Path: " << this->model_save_path_ << std::endl;
    if (is_dir_ok) {
        this->subscribe();
    } else {
        ROS_ERROR("--CANNOT CREATE DIRECTORY..PLEASE CHECK AGAIN");
    }
    this->image_pub_ = this->it_.advertise(
       "/image_annotation/output/segmentation", 1);
}

void ImageAnnotationTool::subscribe() {
    this->image_sub_ = this->it_.subscribe(
       "input", 1,
       &ImageAnnotationTool::imageCb, this);

    this->screenpt_sub_ = this->nh_.subscribe(
        "/camera/rgb/image_rect_color/screenrectangle", 1,
        &ImageAnnotationTool::screenRectangleCb, this);
}

void ImageAnnotationTool::imageCb(
    const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return;
    }
    if ((this->obj_rect_.width > 16 && this->obj_rect_.height > 16) &&
        segment_template_) {
        if (this->mark_model_) {
            this->template_image = cv_ptr->image(this->obj_rect_).clone();
            this->mark_model_ = false;
        }
        if (this->template_image.data) {
            cv::Mat template_img = this->template_image.clone();
            cv::Mat object_roi = this->cvTemplateMatching(
                cv_ptr->image, template_img);
            this->processFrame(cv_ptr->image, object_roi);
        }
    } else {
        this->bootStrapNonObjectDataset(cv_ptr->image, cv::Size(64, 128), 4);
    }
    this->image_pub_.publish(cv_ptr->toImageMsg());
}

void ImageAnnotationTool::screenRectangleCb(
     const geometry_msgs::PolygonStamped::Ptr &pt_msgs) {
     this->obj_rect_.x = pt_msgs->polygon.points[0].x;
     this->obj_rect_.y = pt_msgs->polygon.points[0].y;
     this->obj_rect_.width = pt_msgs->polygon.points[1].x - this->obj_rect_.x;
     this->obj_rect_.height = pt_msgs->polygon.points[1].y - this->obj_rect_.y;
}

cv::Mat ImageAnnotationTool::cvTemplateMatching(
     const cv::Mat &image,
     const cv::Mat &template_img,
     const int match_method) {
     if (image.empty() || template_img.empty()) {
        ROS_ERROR("INPUT IMAGE or TEMPLATE IS EMPTY");
        return cv::Mat();
     }
     int result_cols = image.cols - template_img.cols + sizeof(char);
     int result_rows = image.rows - template_img.rows + sizeof(char);
     cv::Mat result = cv::Mat(result_rows, result_cols, CV_32FC1);
     cv::matchTemplate(image, template_img, result, match_method);
     cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
     double min_val;
     double max_val;
     cv::Point min_loc;
     cv::Point max_loc;
     cv::Point match_loc;
     cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc, cv::Mat());
     match_loc = max_loc;
     const int boundary_padding = 10;
     cv::Rect_<int> rect = cv::Rect_<int>(match_loc.x - boundary_padding,
                                          match_loc.y - boundary_padding,
                                          template_img.cols + boundary_padding,
                                          template_img.rows + boundary_padding);
     if (rect.x + rect.width > image.cols) {
        rect.width -= ((rect.x + rect.width) - image.cols);
     }
     if (rect.y + rect.height > image.rows) {
        rect.height -= ((rect.y + rect.height) - image.rows);
     }
     if (this->iteration_counter_ < 1) {
        iteration_counter_++;
        this->constructColorModel(image, rect, true);
     }
#ifndef DEBUG
     cv::Mat img = image.clone();
     cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);
     cv::imshow("match", img);
     cv::waitKey(3);
#endif
     return image(rect).clone();
}

void ImageAnnotationTool::processFrame(
     const cv::Mat &frame,
     const cv::Mat &image) {
     if (image.empty()) {
        ROS_ERROR("Empty Image ROI can not be segmented");
        return;
     }
     cv::Mat shape_prob = this->shapeProbabilityMap(image);
     cv::Mat color_prob = this->pixelWiseClassification(image);
     cv::Mat mask = this->segmentationProbabilityMap(color_prob, shape_prob);
     cv::Mat model;
     if (mask.data) {
        std::ofstream outFile(
             (this->model_save_path_ + "/" + this->write_txt_name_).c_str(),
             std::ios_base::app);
        cv::Rect_<int> _rect = cv::Rect_<int>(0, 0, image.cols, image.rows);
        model = this->graphCutSegmentation(image, mask, _rect);
        // this->constructColorModel(model, cv::Rect_<int>(), false);
        // this->saveImageModel(
       //    model, "dataset/mask_dataset/drill_", outFile);
       this->saveImageModel(
           image, this->model_save_path_ + "/img_", outFile, true);
       this->model_save_counter_++;
    }
    cv::imshow("model", model);
    cv::imshow("color", color_prob);
    cv::imshow("shape", shape_prob);
    cv::imshow("mask", mask);
    cv::imshow("input", image);
    cv::waitKey(3);
}

cv::Mat ImageAnnotationTool::segmentationProbabilityMap(
    cv::Mat &colorProb,
    cv::Mat &shapeProb) {
    cv::Mat sPM;
    cv::multiply(colorProb, shapeProb, sPM);
    this->cvMorphologicalOperation(sPM);
    return sPM;
}

cv::Mat ImageAnnotationTool::shapeProbabilityMap(const cv::Mat &src) {
    if (src.empty()) {
       ROS_ERROR("Empty Image ROI can not be segmented");
       return cv::Mat();
    }
    cv::Mat image = src.clone();
    if (image.type() != CV_8U) {
       cv::cvtColor(src, image, CV_BGR2GRAY);
    }
    cv::Ptr<cv::FeatureDetector> detector =
       cv::FeatureDetector::create("GFTT");
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(image, keypoints);
    if (keypoints.empty()) {
       ROS_WARN("NO KEYPOINTS AVALIABLE.");
       return cv::Mat();
    }
    std::vector<cv::Point2f> position;
    for (int i = 0; i < keypoints.size(); i++) {
       keypoints[i].convert(keypoints, position);
    }
    cv::Mat hull;
    cv::convexHull(cv::Mat(position), hull);
    cv::Mat shape_prob = cv::Mat::zeros(src.rows, src.cols, CV_32F);
    if (hull.data) {
       for (int j = 0; j < src.rows; j++) {
          for (int i = 0; i < src.cols; i++) {
             cv::Point test_pt = cv::Point(i, j);
             double result = cv::pointPolygonTest(
                cv::Mat(hull), test_pt, false);
             if (result == 1 || result == 0) {
                shape_prob.at<float>(j, i) = 1.0f;
             } else {
                if (shape_prob.at<float>(j, i) != 1) {
                   shape_prob.at<float>(j, i) = 0.0f;
                }
             }
          }
       }
    }
    return shape_prob;
}

cv::Mat ImageAnnotationTool::pixelWiseClassification(
    const cv::Mat &img) {
    cv::Mat likelihood_img = cv::Mat::zeros(img.rows, img.cols, img.type());
    cv::Mat hsvImg;
    cv::cvtColor(img, hsvImg, CV_BGR2HSV);
    std::vector<cv::Mat> split_hsvImg;
    cv::split(hsvImg, split_hsvImg);
    cv::Mat probability = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            int h_pix = static_cast<int>(split_hsvImg[0].at<uchar>(y, x));
            int s_pix = static_cast<int>(split_hsvImg[1].at<uchar>(y, x));
            float p_F = this->fore_histogram_.at<float>(h_pix, s_pix);
            float p_B = this->back_histogram_.at<float>(h_pix, s_pix);
            float color_pro;
            if ((p_F == 0.0f) && (p_B == 0.0f)) {
                color_pro = 1 - 0.0f;
            } else {
               float probability__b = ((p_B * 0.5)/
                                       ((0.5 * p_B) + (0.5 * p_F)));
               color_pro = 1 - probability__b;
            }
            if (color_pro < 1) {
               color_pro = 0;
            } else {
               color_pro = 1;
            }
            probability.at<float>(y, x) = static_cast<float>(color_pro);
            likelihood_img.at<cv::Vec3b>(y, x)[0] = color_pro * 255;
            likelihood_img.at<cv::Vec3b>(y, x)[1] = color_pro * 255;
            likelihood_img.at<cv::Vec3b>(y, x)[2] = color_pro * 255;
        }
    }
    return probability;
}

void ImageAnnotationTool::constructColorModel(
    const cv::Mat &img,
    const cv::Rect &rect,
    bool isbackground) {
    if (!img.data) {
       ROS_ERROR("No image found to compute color model.....");
       return;
    }
    cv::Mat image = img.clone();
    if (!isbackground) {
       this->computeHistogram(image, this->fore_histogram_);
    } else {
       cv::Mat model = img(rect).clone();
       this->computeHistogram(model, this->fore_histogram_);
       rectangle(image, rect, cv::Scalar(0, 0, 0), CV_FILLED);
       this->computeHistogram(image, this->back_histogram_);
    }
    
}

void ImageAnnotationTool::computeHistogram(
    const cv::Mat &src,
    cv::Mat &hist,
    bool isNormalized) {
    int hBin = 180;
    int sBin = 256;
    cv::Mat hsv;
    cv::cvtColor(src, hsv, CV_BGR2HSV);
    int histSize[] = {hBin, sBin};
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    const float* ranges[] = {h_ranges, s_ranges};
    int channels[] = {0, 1};
    calcHist(
       &hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    if (isNormalized) {
       normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    }
}

void ImageAnnotationTool::cvMorphologicalOperation(
    cv::Mat &depth,
    const int erosion_size,
    bool errode) {
    int erosion_type;
    erosion_type = cv::MORPH_RECT;
    cv::Mat erosion_dst;
    cv::Mat element = cv::getStructuringElement(
       erosion_type, cv::Size(2*erosion_size + 1, 2*erosion_size + 1),
       cv::Point(erosion_size, erosion_size));
    if (errode) {
       erode(depth, depth, element);
    } else {
       dilate(depth, depth, element);
    }
}

cv::Mat ImageAnnotationTool::graphCutSegmentation(
    const cv::Mat &image,
    cv::Mat &obj_mask,
    cv::Rect_<int> &rect) {
    int iteration = 1;
    cv::Mat img = image.clone();
    cv::Mat mask = this->createMaskImage(obj_mask);
    cv::Mat fgdModel;
    cv::Mat bgdModel;
    cv::grabCut(img, mask, cv::Rect(), bgdModel, fgdModel,
                static_cast<int>(iteration), cv::GC_INIT_WITH_MASK);
    cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);
    cv::Mat model = cv::Mat::zeros(img.rows, img.cols, img.type());
    this->cvMorphologicalOperation(img, 1);
    img.copyTo(model, mask);
    cv::Rect prevRect = rect;
    rect = this->modelResize(mask);
    if (rect.width < 20 && rect.height < 20) {
        rect = prevRect;
        return img;
    }
    if ((rect.width < (0.7f * prevRect.width) &&
         rect.width > (1.3f * prevRect.width)) ||
        (rect.height < (0.7f * prevRect.height) &&
         rect.height > (1.3f * prevRect.height))) {
       rect = prevRect;
       return img;
    }
     // return model(rect).clone();
    return model;
}


cv::Mat ImageAnnotationTool::createMaskImage(
    cv::Mat &obj_mask) {
    cv::Mat mask = cv::Mat::zeros(obj_mask.rows, obj_mask.cols, CV_8U);
    for (int j = 0; j < obj_mask.rows; j++) {
        for (int i = 0; i < obj_mask.cols; i++) {
           if (obj_mask.at<float>(j, i) <= cv::GC_FGD &&
               obj_mask.at<float>(j,i) > cv::GC_BGD) {
              mask.at<uchar>(j, i) = cv::GC_PR_FGD;
            }
        }
    }
    return mask;
}

cv::Rect_<int> ImageAnnotationTool::modelResize(
    const cv::Mat &src) {
    if (src.empty()) {
       ROS_ERROR("No Image in GraphCutSegmentation::modelResize");
        return cv::Rect_<int>(0, 0, 0, 0);
    }
    cv::Mat src_gray = src.clone();
    if (src_gray.type() != CV_8U) {
       cv::cvtColor(src, src_gray, CV_BGR2GRAY);
    }
    cv::Mat threshold_output;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::threshold(src_gray, threshold_output, 0, 255, CV_THRESH_OTSU);
    findContours(threshold_output, contours, hierarchy,
                 CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
    int area = 0;
    cv::Rect_<int> rect = cv::Rect_<int>();
    for (int i = 0; i< contours.size(); i++) {
       cv::Rect_<int> a = cv::boundingRect(contours[i]);
       if (a.area() > area) {
          area = a.area();
          rect = a;
       }
    }
    return rect;
}

template<typename T>
std::string ImageAnnotationTool::convertNumber2String(
    T c_frame) {
    std::string frame_num;
    std::stringstream out;
    out << c_frame;
    frame_num = out.str();
    return frame_num;
}

void ImageAnnotationTool::saveImageModel(
    const cv::Mat &image, const std::string &spath,
    std::ofstream &outFile, bool ismasked) {
    if (image.data) {
       std::string _save_counter = this->convertNumber2String(
          this->model_save_counter_);
       std::string _save_path;
       std::string write_path;
       if (model_save_counter_ < 10) {
          _save_path = spath + "00000" + _save_counter + ".jpg";
          write_path = "00000" + _save_counter + ".jpg";
       } else if (model_save_counter_ < 100) {
          _save_path = spath + "0000" + _save_counter + ".jpg";
          write_path = "0000" + _save_counter + ".jpg";
       } else if (model_save_counter_ < 1000) {
          _save_path = spath + "000" + _save_counter + ".jpg";
          write_path = "000" + _save_counter + ".jpg";
       } else if (model_save_counter_ < 10000) {
          _save_path = spath + "00" + _save_counter + ".jpg";
          write_path = "00" + _save_counter + ".jpg";
       }
       cv::imwrite(_save_path, image);
       if (ismasked) {
          outFile << write_path << std::endl;
       } else {
           
       }
    }
}


void ImageAnnotationTool::bootStrapNonObjectDataset(
    const cv::Mat &image, const cv::Size wsize, const int incrementer) {
    int icounter = 0;
    std::string _directory =
       "/home/krishneel/catkin_ws/src/image_annotation_tool/";
    std::ofstream nout_file(
       (_directory + "dataset/negative.txt").c_str(), std::ios::out);
    for (int j = 0; j < image.rows; j += incrementer) {
       for (int i = 0; i < image.cols; i += incrementer) {
          cv::Rect_<int> rect = cv::Rect_<int>(i, j, wsize.width, wsize.height);
          if ((rect.x + rect.width < image.cols) &&
              (rect.y + rect.height < image.rows)) {
             cv::Mat roi = image(rect).clone();
             cv::imwrite(_directory + "dataset/negative/image_" +
                         this->convertNumber2String(icounter) + ".jpg", roi);
             nout_file << "dataset/negative/image_"
                       << this->convertNumber2String(icounter++) <<  ".jpg"
                       << std::endl;
          }
       }
    }
    nout_file.close();
}

bool ImageAnnotationTool::directoryExists(
    const char* pzPath) {
    if (pzPath == NULL) {
        return false;
    }
    DIR *pDir;
    bool bExists = false;
    pDir = opendir(pzPath);
    if (pDir != NULL) {
        bExists = true;
        (void)closedir(pDir);
    }
    return bExists;
}

bool ImageAnnotationTool::saveDirectoryHandler(
    const std::string folder_name, std::string &save_folder) {
    if (!this->directoryExists(folder_name.c_str())) {
        boost::filesystem::path dir(folder_name.c_str());
        try {
            boost::filesystem::create_directories(dir);
        } catch(...) {
            ROS_ERROR("--CANNOT CREATE DIRECTORY..\n");
        }
    }
    save_folder = folder_name + "/" + this->timeToStr("");
    boost::filesystem::path ndir(save_folder.c_str());
    return boost::filesystem::create_directories(ndir);
}

template<class T>
std::string ImageAnnotationTool::timeToStr(T ros_t) {
    std::stringstream msg;
    const boost::posix_time::ptime now =
    boost::posix_time::second_clock::local_time();
    boost::posix_time::time_facet *const f =
        new boost::posix_time::time_facet("%Y-%m-%d-%H-%M-%S");
    msg.imbue(std::locale(msg.getloc(), f));
    msg << now;
    return msg.str();
}

int main(int argc, char *argv[]) {

    ros::init(argc, argv, "image_annotation_tool");
    ROS_INFO("RUNNING IMAGE_ANNOTATION_NODELET");
    // std::string _directory = "/home/krishneel/Desktop/";
    ImageAnnotationTool iATool(""/*_directory*/);
    ros::spin();
    return 0;
}
