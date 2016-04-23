// Copyright (C) 2015 by Krishneel Chaudhary,
// JSK Lab, The University of Tokyo

#include <particle_filter_tracking/particle_filter_tracking.h>

ParticleFilterTracking::ParticleFilterTracking():
    block_size_(8), hbins(10), sbins(12), downsize_(1),
    tracker_init_(false), threads_(8) {

    this->hog_ = boost::shared_ptr<HOGFeatureDescriptor>(
       new HOGFeatureDescriptor());

     this->onInit();
}

void ParticleFilterTracking::onInit() {
     this->subscribe();
     this->pub_image_ = pnh_.advertise<sensor_msgs::Image>(
         "target", 1);
}

void ParticleFilterTracking::subscribe() {
     this->sub_screen_pt_ = this->pnh_.subscribe(
         "input_screen", 1, &ParticleFilterTracking::screenPtCB, this);
     this->sub_image_ = this->pnh_.subscribe(
         "image", 1, &ParticleFilterTracking::imageCB, this);
}

void ParticleFilterTracking::unsubscribe() {
     this->sub_image_.shutdown();
}

void ParticleFilterTracking::screenPtCB(
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

void ParticleFilterTracking::imageCB(
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
        this->initializeTracker(image, this->screen_rect_);
        this->tracker_init_ = false;
        this->prev_frame_ = image(screen_rect_).clone();
        ROS_INFO("Tracker Initialization Complete");
     }

     if (this->screen_rect_.width > this->block_size_) {
        this->runObjectTracker(&image, this->screen_rect_);
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

void ParticleFilterTracking::initializeTracker(
     const cv::Mat &image, cv::Rect &rect) {
     this->random_num_ = cv::RNG();
     this->dynamics = this->state_transition();
     this->particles_ = this->initialize_particles(
        this->random_num_, rect.x , rect.y,
        rect.x + rect.width, rect.y + rect.height);

     this->createParticlesFeature(this->reference_features_,
                                  image, particles_);
     this->prev_frame_ = image.clone();
     this->width_ = rect.width;
     this->height_ = rect.height;
}

bool ParticleFilterTracking::createParticlesFeature(
     PFFeatures &features, const cv::Mat &img,
     const std::vector<Particle> &particles) {
     if (img.empty() || particles.empty()) {
         return false;
     }

     const int LENGHT = static_cast<int>(particles.size());
     const int dim = 8/downsize_;
     cv::Mat image;
     image = img.clone();
     cv::cvtColor(img, image, CV_BGR2HSV);
     const int bin_size = 16;

     // cv::Mat histogram = cv::Mat(LENGHT, bin_size * 6, CV_32F);
     cv::Mat histogram[LENGHT];
     cv::Mat hog_descriptors;
     cv::Rect_<int> tmp_rect[LENGHT];

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
     for (int i = 0; i < particles.size(); i++) {
         cv::Rect_<int> rect = cv::Rect_<int>(particles[i].x - dim/2,
                                              particles[i].y - dim/2,
                                              dim, dim);

         this->roiCondition(rect, image.size());
         cv::Mat roi = image(rect).clone();
         cv::Mat part_hist;
         this->getHistogram(part_hist, roi, bin_size, 3, false);

         cv::Point2i s_pt = cv::Point2i((particles[i].x - dim),
                                        (particles[i].y - dim));

         for (int j = 0; j < 2; j++) {
             for (int k = 0; k < 2; k++) {
                 cv::Rect_<int> s_rect = cv::Rect_<int>(
                    s_pt.x, s_pt.y, dim, dim);
                 this->roiCondition(s_rect, image.size());
                 roi = image(s_rect).clone();
                 cv::Mat hist;
                 this->getHistogram(hist, roi, bin_size, 3, false);

                 if (hist.cols == part_hist.cols) {
                     for (int x = 0; x < hist.cols; x++) {
                         part_hist.at<float>(0, x) += hist.at<float>(0, x);
                     }
                 }
                 s_pt.x += dim;
             }
             s_pt.x = particles[i].x - dim;
             s_pt.y += dim;
         }
         s_pt = cv::Point2i((particles[i].x - dim), (particles[i].y - dim));
         rect = cv::Rect_<int>(s_pt.x, s_pt.y, dim * 2, dim * 2);
         this->roiCondition(rect, image.size());
         roi = image(rect).clone();
         cv::Mat region_hist;
         this->getHistogram(region_hist, roi, bin_size, 3, false);
         // cv::normalize(part_hist, part_hist, 0, 1, cv::NORM_MINMAX, -1);

         cv::Mat hist = cv::Mat::zeros(
             1, region_hist.cols + part_hist.cols, CV_32F);
         for (int x = 0; x < part_hist.cols; x++) {
             hist.at<float>(0, x) += part_hist.at<float>(0, x);
         }
         for (int x = part_hist.cols; x < hist.cols; x++) {
             hist.at<float>(0, x) += region_hist.at<float>(
                0, x - part_hist.cols);
         }
         cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1);
         histogram[i] = hist;

         tmp_rect[i] = rect;
     }


     cv::Mat color_hist;
     for (int i = 0; i < particles.size(); i++) {
         cv::Mat roi = img(tmp_rect[i]).clone();
         cv::Mat desc = this->hog_->computeHOG(roi);
         hog_descriptors.push_back(desc);
         color_hist.push_back(histogram[i]);
     }

     features.hog_hist = hog_descriptors;
     features.color_hist = color_hist;
     return true;
}

void ParticleFilterTracking::getHistogram(
     cv::Mat &histogram, const cv::Mat &image,
     const int bins, const int chanel, bool is_norm) {
     if (image.empty()) {
         return;
     }
     histogram = cv::Mat::zeros(sizeof(char), bins * chanel, CV_32F);
     int bin_range = std::ceil(256/bins);

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int j = 0; j < image.rows; j++) {
        for (int i = 0; i < image.cols; i++) {
            float pixel = static_cast<float>(image.at<cv::Vec3b>(j, i)[0]);
            int bin_number = static_cast<int>(std::floor(pixel/bin_range));
            histogram.at<float>(0, bin_number)++;
            
            pixel = static_cast<float>(image.at<cv::Vec3b>(j, i)[1]);
            bin_number = static_cast<int>(std::floor(pixel/bin_range));
            histogram.at<float>(0, bin_number + bins)++;

            pixel = static_cast<float>(image.at<cv::Vec3b>(j, i)[2]);
            bin_number = static_cast<int>(std::floor(pixel/bin_range));
            histogram.at<float>(0, bin_number + bins + bins)++;

            // std::cout << bin_number << ", " << i << " "<< j  << "\n";
        }
    }
    if (is_norm) {
        cv::normalize(histogram, histogram, 0, 1, cv::NORM_MINMAX, -1);
    }
}


void ParticleFilterTracking::runObjectTracker(
    cv::Mat *img, cv::Rect &rect) {
    cv::Mat image = img->clone();
    if (image.empty()) {
       ROS_ERROR("NO IMAGE FRAME TO TRACK");
       return;
    }
    std::vector<Particle> x_particle = this->transition(
       this->particles_, this->dynamics, this->random_num_);
    /*
    this->printParticles(image, x_particle);
    cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
    cv::imshow("Tracking", image);
    return;
    */
    
    std::clock_t start;
    start = std::clock();

    PFFeatures features;
    this->createParticlesFeature(features, image, x_particle);
    std::vector<double> color_probability = this->featureHistogramLikelihood(
       x_particle, image, features, this->reference_features_);

    double duration = (std::clock() - start) /
       static_cast<double>(CLOCKS_PER_SEC);
    std::cout << "Likelihood: "<< duration <<'\n';
    
    std::vector<double> wN;
    for (int i = 0; i < NUM_PARTICLES; i++) {
      double probability = static_cast<double>(
          color_probability[i]);
        wN.push_back(probability);
    }
    std::vector<double> nWeights = this->normalizeWeight(wN);
    this->reSampling(this->particles_, x_particle, nWeights);

    
    this->printParticles(image, particles_);
    Particle x_e = this->meanArr(this->particles_);
    
    cv::Rect b_rect = cv::Rect(
       x_e.x - rect.width/2, x_e.y - rect.height/2,
       this->width_, this->height_);
    cv::circle(image, cv::Point2f(x_e.x, x_e.y), 3,
               cv::Scalar(255, 0, 0), CV_FILLED);
    
    // cv::rectangle(image, b_rect, cv::Scalar(255, 0, 255), 2);
    rect = b_rect;
    cv::resize(image, image, cv::Size(
                  image.cols * downsize_, image.rows * downsize_));
    cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
    cv::imshow("Tracking", image);
}

template<typename T>
T ParticleFilterTracking::EuclideanDistance(
    Particle a, Particle b, bool is_square) {
    T dist = std::pow((a.x - b.x), 2) + std::pow((a.y - b.y), 2);
    if (is_square) {
        dist = std::sqrt(dist);
    }
    return dist;
}


std::vector<double> ParticleFilterTracking::featureHistogramLikelihood(
    const std::vector<Particle> &particles, cv::Mat &image,
    const PFFeatures features, const PFFeatures prev_features) {
    if (features.color_hist.cols != prev_features.color_hist.cols ||
        particles.empty()) {
        return std::vector<double>();
    }

    // cv::Mat results;
    // this->intensityCorrelation(results, image, this->prev_frame_);
    // cv::imshow("results", results);
    
    
    std::vector<double> probability(static_cast<int>(
                                       features.color_hist.rows));
    double *p = &probability[0];
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int i = 0; i < features.color_hist.rows; i++) {
        cv::Mat p_color = features.color_hist.row(i);
        cv::Mat p_hog = features.hog_hist.row(i);
        double c_dist = DBL_MAX;
        double h_dist = DBL_MAX;
        int match_idx = -1;
        
        for (int j = 0; j < prev_features.color_hist.rows; j++) {
            // cv::Mat chist = prev_features.color_hist.row(j);
            // double d_color = cv::compareHist(
            //     chist, p_color, CV_COMP_BHATTACHARYYA);
            // if (d_color < c_dist) {
            //     c_dist = d_color;
            //     match_idx = j;
            // }
            
            cv::Mat hhist = prev_features.hog_hist.row(j);
            double d_hog = cv::compareHist(hhist, p_hog, CV_COMP_BHATTACHARYYA);
            double pt_dist = this->EuclideanDistance<double>(
                this->particles_[j], particles[i]);
            if (d_hog < h_dist) {
                h_dist = d_hog;
                match_idx = j;
            }
        }
        
        double prob = 0.0;
        if (match_idx != -1) {
            // h_dist = cv::compareHist(prev_features.hog_hist.row(match_idx),
            //                          p_hog, CV_COMP_BHATTACHARYYA);
            c_dist = cv::compareHist(prev_features.color_hist.row(match_idx),
                                     p_color, CV_COMP_BHATTACHARYYA);
            double c_prob = 1 * exp(-0.70 * c_dist);
            double h_prob = 1 * exp(-0.70 * h_dist);
            prob = c_prob * h_prob;

            // std::cout << prob  << "\n";
            
            double val = 0.0;
            if (prob < 0.7) {
                prob = 0.0;
            } else if (prob > 0.9) {
                this->reference_features_.color_hist.row(match_idx) =
                    features.color_hist.row(i);
                this->reference_features_.hog_hist.row(match_idx) =
                    features.hog_hist.row(i);
            } else if (prob > 0.7 && prob < 0.9) {
               const float adapt = prob;
               cv::Mat color_ref = reference_features_.color_hist.row(
                  match_idx);
               cv::Mat hog_ref = reference_features_.hog_hist.row(match_idx);
                for (int y = 0; y < color_ref.cols; y++) {
                    color_ref.at<float>(0, y) *= (adapt);
                    color_ref.at<float>(0, y) += (
                        (1.0f - adapt) * features.color_hist.row(
                           i).at<float>(0, y));
                }
                for (int y = 0; y < hog_ref.cols; y++) {
                    hog_ref.at<float>(0, y) *= (adapt);
                    hog_ref.at<float>(0, y) += (
                        (1.0f- adapt) * features.hog_hist.row(i).at<
                        float>(0, y));
                }
                this->reference_features_.color_hist.row(match_idx) = color_ref;
                this->reference_features_.hog_hist.row(match_idx) = hog_ref;
            }
            // std::cout << "Prob: " << p[i] << " " << val << " " << p[i]
            // * val  << "\n";
        }
        p[i] = prob;
    }
    cv::namedWindow("Ref", cv::WINDOW_NORMAL);
    cv::imshow("Ref", reference_features_.color_hist);

    return probability;
}

void ParticleFilterTracking::roiCondition(
    cv::Rect &rect, cv::Size imageSize) {
    if (rect.x < 0) {
        rect.x = 0;
        // rect.width = block_size_;
    }
    if (rect.y < 0) {
        rect.y = 0;
        // rect.height = block_size_;
    }
    if ((rect.height + rect.y) > imageSize.height) {
        rect.y -= ((rect.height + rect.y) - imageSize.height);
        // rect.height = block_size_;
    }
    if ((rect.width + rect.x) > imageSize.width) {
        rect.x -= ((rect.width + rect.x) - imageSize.width);
        // rect.width = block_size_;
    }
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "gpu_particle_filter");
    ParticleFilterTracking pfg;
    ros::spin();
    return 0;
}
