
#include <particle_filter_tracking/particle_filter_tracking.h>

ParticleFilterTracking::ParticleFilterTracking() :
    block_size_(16), hbins(10), sbins(12), tracker_init_(false) {
    this->subscribe();
    this->onInit();
}

void ParticleFilterTracking::onInit() {
    this->pub_image_ = pnh_.advertise<sensor_msgs::Image>(
       "target", 1);
}

void ParticleFilterTracking::subscribe() {
    this->sub_screen_pt_ = this->pnh_.subscribe(
       "input_screen", 1, &ParticleFilterTracking::screenPointCallback, this);
    this->sub_image_ = this->pnh_.subscribe(
       "input", 1, &ParticleFilterTracking::imageCallback, this);
}

void ParticleFilterTracking::unsubscribe() {
    this->sub_image_.shutdown();
}

void ParticleFilterTracking::screenPointCallback(
    const geometry_msgs::PolygonStamped &screen_msg) {
    int x = screen_msg.polygon.points[0].x;
    int y = screen_msg.polygon.points[0].y;
    int width = screen_msg.polygon.points[1].x - x;
    int height = screen_msg.polygon.points[1].y - y;
    this->screen_rect_ = cv::Rect_<int>(x, y, width, height);
    if (width > this->block_size_ && height > this->block_size_) {
       this->tracker_init_ = true;
    } else {
       ROS_WARN("-- Selected Object Size is too small... Not init tracker");
    }
}

void ParticleFilterTracking::imageCallback(
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

    if (this->tracker_init_) {
       ROS_INFO("Initializing Tracker");
       this->initializeTracker(image, this->screen_rect_);
       this->tracker_init_ = false;
       this->runObjectTracker(&image, this->screen_rect_);
    }

    if (this->screen_rect_.width > this->block_size_) {
       ROS_INFO("Running Tracker..");
       this->runObjectTracker(&image, this->screen_rect_);
    }
    
    cv_bridge::CvImagePtr pub_msg(new cv_bridge::CvImage);
    pub_msg->header = image_msg->header;
    pub_msg->encoding = sensor_msgs::image_encodings::BGR8;
    pub_msg->image = image.clone();
    this->pub_image_.publish(pub_msg);

    cv::waitKey(3);
}

void ParticleFilterTracking::initializeTracker(
    const cv::Mat &image, cv::Rect &rect) {
    this->randomNum = cv::RNG();
    this->dynamics = this->state_transition();
    this->particles = this->initialize_particles(
       this->randomNum, rect.x , rect.y,
       rect.x + rect.width, rect.y + rect.height);
    cv::Mat object_region = image(rect).clone();
    cv::Mat scene_region = image.clone();
    cv::rectangle(scene_region, rect, cv::Scalar(0, 0, 0), CV_FILLED);
    this->reference_object_histogram_ = this->imagePatchHistogram(
       object_region);
    this->reference_background_histogram_ = this->imagePatchHistogram(
       scene_region);
    this->particle_prev_position.clear();
    for (int i = 0; i < NUM_PARTICLES; i++) {
        this->particle_prev_position.push_back(
           cv::Point2f(this->particles[i].x, this->particles[i].y));
    }
    this->prevFrame = image.clone();
    this->width_ = rect.width;
    this->height_ = rect.height;
}

void ParticleFilterTracking::runObjectTracker(
    cv::Mat *img, cv::Rect &rect) {
    cv::Mat image = img->clone();
    if (image.empty()) {
       ROS_ERROR("NO IMAGE FRAME TO TRACK");
       return;
    }
    std::vector<Particle> x_particle = this->transition(
       this->particles, this->dynamics, this->randomNum);
    this->prevPts.clear();
    for (int i = 0; i < NUM_PARTICLES; i++) {
       this->prevPts.push_back(cv::Point2f(x_particle[i].x, x_particle[i].y));
    }
    this->getOpticalFlow(image, this->prevFrame, this->prevPts);
    std::vector<double> of_velocity;
    for (int i = 0; i < this->prevPts.size(); i++) {
        double vel_X = this->prevPts[i].x - particle_prev_position[i].x;
        double vel_Y = this->prevPts[i].y - particle_prev_position[i].y;
        double vel = sqrt((vel_X * vel_X) + (vel_Y * vel_Y));
        of_velocity.push_back(vel);
    }
    std::vector<cv::Mat> particle_histogram = this->particleHistogram(
       image, x_particle);
    std::vector<double> color_probability = this->colorHistogramLikelihood(
       particle_histogram);
    std::vector<double> motion_probability = this->motionLikelihood(
       of_velocity, x_particle, particles);
    std::vector<double> wN;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        double probability = static_cast<double>(
           color_probability[i] * motion_probability[i]);
        wN.push_back(probability);
    }
    std::vector<double> nWeights = this->normalizeWeight(wN);
    this->reSampling(this->particles, x_particle, nWeights);

    this->printParticles(image, particles);
    
    Particle x_e = this->meanArr(this->particles);
    cv::Rect b_rect = cv::Rect(
       x_e.x - rect.width/2, x_e.y - rect.height/2,
       this->width_, this->height_);
    cv::circle(image, cv::Point2f(x_e.x, x_e.y), 3,
               cv::Scalar(255, 0, 0), CV_FILLED);
    cv::rectangle(image, b_rect, cv::Scalar(255, 0, 255), 2);
    rect = b_rect;
    imshow("Environmental Tracking", image);
}

std::vector<double> ParticleFilterTracking::colorHistogramLikelihood(
    std::vector<cv::Mat> &obj_patch) {
    std::vector<double> prob_hist;
#pragma omp parallel for
    for (int i = 0; i < obj_patch.size(); i++) {
       double dist = static_cast<double>(
          this->computeHistogramDistances(
             obj_patch[i], &this->reference_object_histogram_));
       double bg_dist = static_cast<double>(
          this->computeHistogramDistances(
             obj_patch[i], &this->reference_background_histogram_));
        double pr = 0.0;
        if (dist < bg_dist) {
            pr = 2 * exp(-2 * dist);
        } else {
            pr = 0.0;
        }
        prob_hist.push_back(pr);
    }
    return prob_hist;
}

double ParticleFilterTracking::computeHistogramDistances(
    cv::Mat &hist, std::vector<cv::Mat> *hist_MD , cv::Mat *h_D ) {
    double sum = 0.0;
    double argMaxDistance = 100.0;
    if (h_D != NULL) {
       sum = static_cast<double>(
          cv::compareHist(hist, *h_D, CV_COMP_BHATTACHARYYA));
    } else if (hist_MD->size() > 0) {
#pragma omp parallel for
       for (int i = 0; i < hist_MD->size(); i++) {
          double d__ = static_cast<double>(
             cv::compareHist(hist, (*hist_MD)[i], CV_COMP_BHATTACHARYYA));
          if (d__ < argMaxDistance) {
             argMaxDistance = static_cast<double>(d__);
          }
       }
       sum = static_cast<double>(argMaxDistance);
    }
    return sum;
}

std::vector<cv::Mat> ParticleFilterTracking::particleHistogram(
    cv::Mat &image, std::vector<Particle> &p) {
    std::vector<cv::Mat> obj_histogram;
    if (image.empty()) {
       return std::vector<cv::Mat>();
    }
    for (int i = 0; i < NUM_PARTICLES; i++) {
       cv::Rect p_rect = cv::Rect(p[i].x - block_size_/2,
                                  p[i].y - block_size_/2,
                                  block_size_,
                                  block_size_);
       roiCondition(p_rect, image.size());
       cv::Mat roi = image(p_rect).clone();
       cv::Mat h_D;
       this->computeHistogram(roi, h_D, true);
       obj_histogram.push_back(h_D);
    }
    return obj_histogram;
}

std::vector<cv::Mat> ParticleFilterTracking::imagePatchHistogram(
    cv::Mat &image) {
    if (image.empty()) {
       return std::vector<cv::Mat>();
    }
    const int OVERLAP = 2;
    std::vector<cv::Mat> _patch_hist;
    for (int j = 0; j < image.rows; j += (block_size_/OVERLAP)) {
       for (int i = 0; i < image.cols; i += (block_size_/OVERLAP)) {
           cv::Rect rect = cv::Rect(i, j, block_size_, block_size_);
           roiCondition(rect, image.size());
           cv::Mat roi = image(rect);
           cv::Mat h_MD;
           this->computeHistogram(roi, h_MD, true);
           _patch_hist.push_back(h_MD);
        }
    }
    return _patch_hist;
}

double ParticleFilterTracking::gaussianNoise(
    double a, double b) {
    return rand() / (RAND_MAX + 1.0) * (b - a) + a;
}

double ParticleFilterTracking::motionVelocityLikelihood(
    double distance) {
    double pr = ((1 - gaussianNoise(0, 1)) * exp(-1 * distance)) +
       gaussianNoise(0, 1);
    return pr;
}

cv::Point2f ParticleFilterTracking::motionCovarianceEstimator(
    std::vector<cv::Point2f> &prevPts, std::vector<Particle> &x_e) {
    double sumX = 0.0;
    double sumY = 0.0;
    for (int i = 0; i < prevPts.size(); i++) {
        double v_X = static_cast<double>(abs(x_e[i].dx - prevPts[i].x));
        double v_Y = static_cast<double>(abs(x_e[i].dy - prevPts[i].y));
        sumX += (v_X * v_X);
        sumY += (v_Y * v_Y);
    }
    double varianceX = static_cast<double>(sumX / (prevPts.size() - 1));
    double varianceY = static_cast<double>(sumY / (prevPts.size() - 1));
    return cv::Point2f(varianceX, varianceY);
}

std::vector<double> ParticleFilterTracking::motionLikelihood(
    std::vector<double> &prevPts, std::vector<Particle> &x_particle,
    std::vector<Particle> &particles) {
    std::vector<double> m_prob;
    std::vector<double> particle_vel;
    for (int i = 0; i < x_particle.size(); i++) {
       double x = std::sqrt((x_particle[i].dx*x_particle[i].dx) +
                            (x_particle[i].dy*x_particle[i].dy));
        particle_vel.push_back(x);
    }
    for (int i = 0; i < prevPts.size(); i++) {
        double v_X = static_cast<double>(particle_vel[i] - prevPts[i]);
        double d_x = static_cast<double>(motionVelocityLikelihood(abs(v_X)));
        m_prob.push_back(d_x);
    }
    return m_prob;
}

void ParticleFilterTracking::roiCondition(
    cv::Rect &rect, cv::Size imageSize) {
    if (rect.x < 0) {
        rect.x = 0;
        rect.width = block_size_;
    }
    if (rect.y < 0) {
        rect.y = 0;
        rect.height = block_size_;
    }
    if ((rect.height + rect.y) > imageSize.height) {
        rect.y -= ((rect.height + rect.y) - imageSize.height);
        rect.height = block_size_;
    }
    if ((rect.width + rect.x) > imageSize.width) {
        rect.x -= ((rect.width + rect.x) - imageSize.width);
        rect.width = block_size_;
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "particle_filter_tracking");
    ParticleFilterTracking pft;
    ros::spin();
    return 0;
}
