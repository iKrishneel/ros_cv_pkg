
#include <kernelized_correlation_filters/kcf.h>
#include <numeric>
// #include <future>
// #include <thread>

KCF_Tracker::KCF_Tracker() {
    p_resize_image = false;
    p_padding = 1.5;
    p_output_sigma_factor = 0.1;
    p_output_sigma;
    p_kernel_sigma = 0.5;    // def = 0.5
    p_lambda = 1e-4;         // regularization in learning step
    p_interp_factor = 0.02;
    p_cell_size = 4;
    p_scale_step = 1.02;
    p_current_scale = 1.0f;
    p_num_scales = 7;
    FILTER_SIZE_ = 0;
    FILTER_BATCH_ = 256;

    is_cnn_set_ = false;
    m_use_scale = !true;
    m_use_color = true;
    m_use_subpixel_localization = true;
    m_use_subgrid_scale = true;
    m_use_multithreading = true;
    m_use_cnfeat = true;

    init_cufft_plan_ = true;
    
}

KCF_Tracker::KCF_Tracker(
    double padding, double kernel_sigma, double lambda,
    double interp_factor, double output_sigma_factor,
    int cell_size) :
    p_padding(padding),
    p_output_sigma_factor(output_sigma_factor),
    p_kernel_sigma(kernel_sigma),
    p_lambda(lambda),
    p_interp_factor(interp_factor),
    p_cell_size(cell_size),
    is_cnn_set_(false) {

    /*
      padding      ... extra area surrounding the target           (1.5)
      kernel_sigma ... gaussian kernel bandwidth                   (0.5)
      lambda       ... regularization                              (1e-4)
      interp_factor ... linear interpolation factor for adaptation  (0.02)
      output_sigma_factor... spatial bandwidth (proportional to target)  (0.1)
      cell_size       ... hog cell size                               (4)
    */
}

void KCF_Tracker::setCaffeInfo(
    const std::string pretrained_weights, const std::string model_prototxt,
    const std::string mean_file,
    std::vector<std::string> &feature_layers, const int device_id) {
    this->feature_extractor_ = new FeatureExtractor(
       pretrained_weights, model_prototxt, mean_file,
       feature_layers, device_id);
    this->is_cnn_set_ = true;
}


void KCF_Tracker::init(cv::Mat &img, const cv::Rect & bbox) {
    // check boundary, enforce min size
    if (!this->is_cnn_set_) {
       ROS_FATAL("CAFFE CNN INFO NOT SET");
       return;
    }
    double x1 = bbox.x;
    double x2 = bbox.x + bbox.width;
    double y1 = bbox.y;
    double y2 = bbox.y + bbox.height;
    // if (x1 < 0) x1 = 0.;
    // if (x2 > img.cols-1) x2 = img.cols - 1;
    // if (y1 < 0) y1 = 0;
    // if (y2 > img.rows-1) y2 = img.rows - 1;

    x1 = (x1 < 0.0) ? 0.0 : x1;
    x2 = (x2 > img.cols - 1) ? img.cols - 1 : x2;
    y1 = (y1 < 0.0) ? 0.0 : y1;
    y2 = (y2 > img.rows - 1) ? img.rows - 1 : y2;
    
    if (x2 - x1 < 2 * p_cell_size) {
        double diff = (2 * p_cell_size - x2 + x1) / 2.0;
        if (x1 - diff >= 0 && x2 + diff < img.cols) {
            x1 -= diff;
            x2 += diff;
        } else if (x1 - 2*diff >= 0) {
            x1 -= 2*diff;
        } else {
            x2 += 2*diff;
        }
    }
    if (y2 - y1 < 2 * p_cell_size) {
        double diff = (2*p_cell_size -y2+y1)/2.;
        if (y1 - diff >= 0 && y2 + diff < img.rows) {
            y1 -= diff;
            y2 += diff;
        } else if (y1 - 2*diff >= 0) {
            y1 -= 2*diff;
        } else {
            y2 += 2*diff;
        }
    }

    p_pose.w = x2-x1;
    p_pose.h = y2-y1;
    p_pose.cx = x1 + p_pose.w/2.;
    p_pose.cy = y1 + p_pose.h/2.;

    cv::Mat input_gray, input_rgb = img.clone();
    if (img.channels() == 3) {
       cv::cvtColor(img, input_gray, CV_BGR2GRAY);
       input_gray.convertTo(input_gray, CV_32FC1);
    } else {
       img.convertTo(input_gray, CV_32FC1);
    }
    
    // don't need too large image
    if (p_pose.w * p_pose.h > 100.* 100.) {
        std::cout << "resizing image by factor of 2" << std::endl;
        p_resize_image = true;
        p_pose.scale(0.5);
        cv::resize(input_gray, input_gray, cv::Size(0, 0),
                   0.5, 0.5, cv::INTER_AREA);
        cv::resize(input_rgb, input_rgb, cv::Size(0, 0),
                   0.5, 0.5, cv::INTER_AREA);
    }

    // compute win size + fit to fhog cell size
    p_windows_size[0] = round(p_pose.w * (1. + p_padding) /
                              p_cell_size) * p_cell_size;
    p_windows_size[1] = round(p_pose.h * (1. + p_padding) /
                              p_cell_size) * p_cell_size;
    
    p_scales.clear();
    if (m_use_scale) {
        for (int i = -p_num_scales/2; i <= p_num_scales/2; ++i)
            p_scales.push_back(std::pow(p_scale_step, i));
    } else {
        p_scales.push_back(1.);
    }
    
    p_current_scale = 1.;

    double min_size_ratio = std::max(5.0f * p_cell_size/ p_windows_size[0],
                                     5.0f * p_cell_size/p_windows_size[1]);
    double max_size_ratio = std::min(
       floor((img.cols + p_windows_size[0]/3)/p_cell_size)*
       p_cell_size/p_windows_size[0],
       floor((img.rows + p_windows_size[1]/3)/p_cell_size)*
       p_cell_size/p_windows_size[1]);
    p_min_max_scale[0] = std::pow(p_scale_step,
                                  std::ceil(std::log(min_size_ratio) /
                                            log(p_scale_step)));
    p_min_max_scale[1] = std::pow(p_scale_step,
                                  std::floor(std::log(max_size_ratio) /
                                             log(p_scale_step)));

    std::cout << "init: img size " << img.cols << " " << img.rows << std::endl;
    std::cout << "init: win size. " << p_windows_size[0] << " "
              << p_windows_size[1] << std::endl;
    std::cout << "init: min max scales factors: " << p_min_max_scale[0]
              << " " << p_min_max_scale[1] << std::endl;

    p_output_sigma = std::sqrt(p_pose.w*p_pose.h) *
       p_output_sigma_factor / static_cast<double>(p_cell_size);

    
    // window weights, i.e. labels
    cv::Mat gsl = gaussian_shaped_labels(p_output_sigma,
                                         p_windows_size[0]/p_cell_size,
                                         p_windows_size[1]/p_cell_size);

    this->FILTER_SIZE_ = gsl.rows * gsl.cols;
    
    //! setup cuffthandles
    cufftResult cufft_status;
    cufft_status = cufftPlan1d(&cufft_handle1_, FILTER_SIZE_,
                               CUFFT_C2C, 1);
    if (cufft_status != cudaSuccess) {
       ROS_FATAL("CUDA FFT HANDLE CREATION FAILED");
       std::exit(1);
    }

    cufft_status = cufftPlan1d(
       &handle_, FILTER_SIZE_, CUFFT_C2C, FILTER_BATCH_);
    if (cufft_status != cudaSuccess) {
       ROS_ERROR("CUDAFFT PLAN [C2C] ALLOC FAILED");
       std::exit(-1);  //! change to shutdown
    }
    cufft_status = cufftPlan1d(
       &inv_handle_, FILTER_SIZE_, CUFFT_C2R, FILTER_BATCH_);
    if (cufft_status != cudaSuccess) {
       ROS_ERROR("CUDAFFT PLAN [C2R] ALLOC FAILED");
       std::exit(-1);  //! change to shutdown
    }

    cufft_status = cufftPlan1d(
       &inv_cufft_handle1_, FILTER_SIZE_, CUFFT_C2R, 1);
     if (cufft_status != cudaSuccess) {
        ROS_ERROR("CUDAFFT PLAN [C2R] ALLOC FAILED FOR BATCH = 1");
        exit(1);
     }

     ROS_INFO("\n\033[35m ALL CUFFT PLAN SETUP DONE \033[0m\n");
    
    
    p_yf = fft2(gsl);  //! GPU

    //! fft on gpu

    float *dev_data;
    int IN_BYTE = FILTER_SIZE_ * sizeof(float);
    cudaMalloc(reinterpret_cast<void**>(&dev_data), IN_BYTE);
    cudaMemcpy(dev_data, reinterpret_cast<float*>(gsl.data), IN_BYTE,
               cudaMemcpyHostToDevice);
    cufftComplex *dev_p_yf = convertFloatToComplexGPU(
       dev_data, 1, FILTER_SIZE_);
    
    cufft_status = cufftExecC2C(cufft_handle1_, dev_p_yf,
                                dev_p_yf, CUFFT_FORWARD);

    if (cufft_status != cudaSuccess) {
       ROS_FATAL("CUDA FFT [cufftExecC2C] FAILED");
       cudaFree(dev_p_yf);
       cudaFree(dev_data);
       return;
    }
    
    // p_cos_window = cosine_window_function(p_yf.cols, p_yf.rows);
    p_cos_window = cosine_window_function(gsl.cols, gsl.rows);
    
    // this->FILTER_SIZE_ = p_cos_window.rows * p_cos_window.cols;
    this->BYTE_ = FILTER_BATCH_ * p_cos_window.rows *
       p_cos_window.cols * sizeof(float);
    float *cosine_window_1D = reinterpret_cast<float*>(std::malloc(BYTE_));
    int icounter = 0;
    for (int i = 0; i < FILTER_BATCH_; i++) {
       
       for (int j = 0; j < p_cos_window.rows; j++) {
          for (int k = 0; k < p_cos_window.cols; k++) {
             cosine_window_1D[icounter] = p_cos_window.at<float>(j, k);
             icounter++;
          }
       }
    }
    cudaMalloc(reinterpret_cast<void**>(&d_cos_window_), BYTE_);
    cudaMemcpy(d_cos_window_, cosine_window_1D, BYTE_, cudaMemcpyHostToDevice);
     
    // obtain a sub-window for training initial model
    std::vector<cv::Mat> path_feat = get_features(input_rgb, input_gray,
                                                  p_pose.cx, p_pose.cy,
                                                  p_windows_size[0],
                                                  p_windows_size[1]);
    p_model_xf = fft2(path_feat, p_cos_window);

    // Kernel Ridge Regression, calculate alphas (in Fourier domain)
    ComplexMat kf = gaussian_correlation(p_model_xf, p_model_xf,
                                         p_kernel_sigma, true);


    /*
    //! EXACLTLY SAME DATA
    float c_feat[FILTER_BATCH_ * FILTER_SIZE_ ];
    cufftComplex cv_fft[FILTER_BATCH_ * FILTER_SIZE_ ];
    
    int icount = 0;
    for (int i = 0; i < path_feat.size(); i++) {
       for (int y = 0; y < path_feat[i].rows; y++) {
          for (int x = 0; x < path_feat[i].cols; x++) {
             c_feat[icount] = path_feat[i].at<float>(y,  x);
             icount++;
          }
       }
    }
    std::cout << "total: " << icount << " "
              << FILTER_BATCH_ * FILTER_SIZE_ << "\n";
    
    // float *dev_feat;
    // cudaMalloc(reinterpret_cast<void**>(&dev_feat), BYTE_);
    // cudaMemcpy(dev_feat, c_feat, BYTE_, cudaMemcpyHostToDevice);

    /**
     * start
     */

    /*

    //! obtain a sub-window for training initial model
    // const float *d_model_features
    float *dev_feat = get_featuresGPU(input_rgb, input_gray,
                                      p_pose.cx, p_pose.cy,
                                      p_windows_size[0],
                                      p_windows_size[1], 1.0f);
    // float *dev_feat = const_cast<float*>(d_model_features);
    const int data_lenght = window_size_.width *
       window_size_.height * FILTER_BATCH_;
    
    float *dev_cos = cosineConvolutionGPU(dev_feat, d_cos_window_,
                                          FILTER_BATCH_ * FILTER_SIZE_,
                                          BYTE_);
    // cufftComplex *dev_model_xf_ = this->cuDFT(dev_cos);
    dev_model_xf_ = this->cuDFT(dev_cos);
    
    float kf_xf_norm = 0.0f;
    float *dev_kxyf = squaredNormAndMagGPU(kf_xf_norm, dev_model_xf_,
                                          FILTER_BATCH_, FILTER_SIZE_);

    ROS_INFO("\033[35m INIT NORM: %3.3f \033[0m", kf_xf_norm);
    
    float kf_yf_norm = kf_xf_norm;
    cufftComplex *dev_kxyf_complex = convertFloatToComplexGPU(dev_kxyf,
                                                              FILTER_BATCH_,
                                                              FILTER_SIZE_);
    float *dev_kifft = this->cuInvDFT(dev_kxyf_complex);
    float *dev_xysum = invFFTSumOverFiltersGPU(dev_kifft,
                                               FILTER_BATCH_, FILTER_SIZE_);
    float normalizer = 1.0f / (static_cast<float>(FILTER_SIZE_*FILTER_BATCH_));
    cuGaussianExpGPU(dev_xysum, kf_xf_norm, kf_yf_norm, p_kernel_sigma,
                     normalizer, FILTER_SIZE_);
    cufftComplex *dev_kf = convertFloatToComplexGPU(dev_xysum, 1,
                                                    FILTER_SIZE_);
    cufftExecC2C(cufft_handle1_, dev_kf, dev_kf, CUFFT_FORWARD);

    */
     // p_model_alphaf = p_yf / (kf + p_lambda);   //equation for fast training
    
     p_model_alphaf_num = p_yf * kf;
     p_model_alphaf_den = kf * (kf + p_lambda);
     p_model_alphaf = p_model_alphaf_num / p_model_alphaf_den;


     /*
     //! training on device
     const int dimension = FILTER_SIZE_;
     this->dev_model_alphaf_num_ = multiplyComplexGPU(
        dev_p_yf, dev_kf, dimension);
     cufftComplex *dev_temp = addComplexByScalarGPU(
        dev_kf, static_cast<float>(p_lambda), dimension);
     this->dev_model_alphaf_den_ = multiplyComplexGPU(
        dev_kf, dev_temp, dimension);
     this->dev_model_alphaf_ = divisionComplexGPU(
        this->dev_model_alphaf_num_, this->dev_model_alphaf_den_, dimension);

     /**
      * copy to cpu for debuggging
      */

     /*
     ROS_WARN("GPU NORM: %3.3f", kf_xf_norm);
     ROS_ERROR("PRINTING..");
     cufftComplex exp_data[FILTER_SIZE_];
     cudaMemcpy(exp_data, dev_model_alphaf_,  // FILTER_BATCH_ *
                FILTER_SIZE_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
     std::ofstream outfile("exp.txt");
     for (int i = 0; i < FILTER_SIZE_;  i++) {
        outfile  << exp_data[i].x << " " << exp_data[i].y << "\n";
     }
     outfile.close();
     ROS_WARN("DONE...");
     std::cout << p_model_alphaf  << "\n";
     exit(1);
     */
     
     
     //! clean up
     /*
     cudaFree(dev_cos);
     cudaFree(dev_feat);
     cudaFree(dev_kxyf);
     cudaFree(dev_temp);
     cudaFree(dev_kxyf_complex);
     cudaFree(dev_kf);
     */
     // cudaFree(d_model_features);
     
     // cudaFree(dev_model_xf);
     // cufftDestroy(cufft_handle1_);
     
     free(cosine_window_1D);
}

void KCF_Tracker::setTrackerPose(BBox_c &bbox, cv::Mat & img) {
     init(img, bbox.get_rect());
}

void KCF_Tracker::updateTrackerPosition(BBox_c &bbox) {
    if (p_resize_image) {
       BBox_c tmp = bbox;
       tmp.scale(0.5);
       p_pose.cx = tmp.cx;
       p_pose.cy = tmp.cy;
    } else {
       p_pose.cx = bbox.cx;
       p_pose.cy = bbox.cy;
    }
}

BBox_c KCF_Tracker::getBBox() {
    BBox_c tmp = p_pose;
    tmp.w *= p_current_scale;
    tmp.h *= p_current_scale;

    if (p_resize_image)
       tmp.scale(2);
    
    return tmp;
}

void KCF_Tracker::track(cv::Mat &img) {
     cv::Mat input_gray, input_rgb = img.clone();
     if (img.channels() == 3) {
         cv::cvtColor(img, input_gray, CV_BGR2GRAY);
         input_gray.convertTo(input_gray, CV_32FC1);
     } else {
         img.convertTo(input_gray, CV_32FC1);
     }

     // don't need too large image
     if (p_resize_image) {
         cv::resize(input_gray, input_gray, cv::Size(0, 0),
                    0.5, 0.5, cv::INTER_AREA);
         cv::resize(input_rgb, input_rgb, cv::Size(0, 0),
                    0.5, 0.5, cv::INTER_AREA);
     }

     std::vector<cv::Mat> patch_feat;
     double max_response = -1.;
     cv::Mat max_response_map;
     cv::Point2i max_response_pt;
     int scale_index = 0;
     std::vector<double> scale_responses;

     //! running on gpu
     std::vector<cv::cuda::GpuMat> patch_feat_gpu;
     cv::cuda::GpuMat d_cos_window(p_cos_window);

     /*
     //! test
     cv::Mat igray = input_rgb(cv::Rect(50, 50, 13, 13));
     // cv::resize(input_rgb, igray, cv::Size(13, 13));
     cv::cvtColor(igray, igray, CV_BGR2GRAY);
     igray.convertTo(igray, CV_32FC1);
     float *iresize = bilinear_test(
        reinterpret_cast<float*>(igray.data),
        input_gray.rows * input_gray.step);
     cv::Mat resize_im = cv::Mat::zeros(50, 50, CV_8UC1);
     int icount = 0;
     for (int i = 0; i < resize_im.rows; i++) {
        for (int j = 0; j < resize_im.cols; j++) {
           resize_im.at<uchar>(i, j) = iresize[icount++];
        }
     }
     cv::namedWindow("rimage", CV_WINDOW_NORMAL);
     cv::imshow("rimage", resize_im);
     // cv::imshow("rimage", igray);
     cv::waitKey(3);
     return;
     */ 

     for (size_t i = 0; i < p_scales.size(); ++i) {

        
        std::clock_t start;
        double duration;
        start = std::clock();
        /**
         * GPU --------------------------------------
         */
        ROS_INFO("---------RUNNING GPU---------");
        /*
        float *d_features = get_featuresGPU(
           input_rgb, input_gray, p_pose.cx, p_pose.cy, p_windows_size[0],
           p_windows_size[1], p_current_scale * p_scales[i]);

        cv::Mat response = this->trackingProcessOnGPU(d_features);
        
        duration = (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
        std::cout << " GPU PROCESS: : " << duration <<'\n';
        */
        /**
         * END GPU ----------------------------------
         */


        /**
         * debug cuda functions
         */
        


        patch_feat = get_features(
           input_rgb, input_gray, p_pose.cx, p_pose.cy, p_windows_size[0],
           p_windows_size[1], p_current_scale * p_scales[i]);


        // float c_feat[FILTER_BATCH_ * FILTER_SIZE_];
        // int icount = 0;
        // for (int i = 0; i < patch_feat.size(); i++) {
        //    for (int y = 0; y < patch_feat[i].rows; y++) {
        //       for (int x = 0; x < patch_feat[i].cols; x++) {
        //          c_feat[icount] = patch_feat[i].at<float>(y,  x);
        //          icount++;
        //       }
        //    }
        // }
        // float *d_features;
        // cudaMalloc(reinterpret_cast<void**>(&d_features), BYTE_);
        // cudaMemcpy(d_features, c_feat, BYTE_, cudaMemcpyHostToDevice);
        
        

        ComplexMat zf = fft2(patch_feat, p_cos_window);
        ComplexMat kzf = gaussian_correlation(zf, p_model_xf,
                                              p_kernel_sigma);
        
        cv::Mat response = ifft2(p_model_alphaf * kzf);

        // target location is at the maximum response. we must take into
        //    account the fact that, if the target doesn't move, the peaka
        //    will appear at the top-left corner, not at the center (this is
        //    discussed in the paper). the responses wrap around
        //    cyclically.
        
        double min_val, max_val;
        cv::Point2i min_loc, max_loc;
        cv::minMaxLoc(response, &min_val, &max_val, &min_loc, &max_loc);

        
        duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
        std::cout << " CPU PROCESS: : " << duration <<'\n';
        



        double weight = p_scales[i] < 1. ? p_scales[i] : 1./p_scales[i];
        if (max_val*weight > max_response) {
           max_response = max_val*weight;
           max_response_map = response;
           max_response_pt = max_loc;
           scale_index = i;
        }
        scale_responses.push_back(max_val*weight);


     }


     // sub pixel quadratic interpolation from neighbours
     // wrap around to negative half-space of vertical axis
     if (max_response_pt.y > max_response_map.rows / 2) {
         max_response_pt.y = max_response_pt.y - max_response_map.rows;
     }
     // same for horizontal axis
     if (max_response_pt.x > max_response_map.cols / 2) {
         max_response_pt.x = max_response_pt.x - max_response_map.cols;
     }

     cv::Point2f new_location(max_response_pt.x, max_response_pt.y);

     if (m_use_subpixel_localization) {
         new_location = sub_pixel_peak(max_response_pt, max_response_map);
     }

     p_pose.cx += p_current_scale*p_cell_size*new_location.x;
     p_pose.cy += p_current_scale*p_cell_size*new_location.y;
     if (p_pose.cx < 0) p_pose.cx = 0;
     if (p_pose.cx > img.cols-1) p_pose.cx = img.cols-1;
     if (p_pose.cy < 0) p_pose.cy = 0;
     if (p_pose.cy > img.rows-1) p_pose.cy = img.rows-1;

     // sub grid scale interpolation
     double new_scale = p_scales[scale_index];
     if (m_use_subgrid_scale)
         new_scale = sub_grid_scale(scale_responses, scale_index);

     p_current_scale *= new_scale;

     if (p_current_scale < p_min_max_scale[0])
         p_current_scale = p_min_max_scale[0];
     if (p_current_scale > p_min_max_scale[1])
         p_current_scale = p_min_max_scale[1];

     // obtain a subwindow for training at newly estimated target
     // position
     bool is_update = false;
     if (is_update) {
        patch_feat = get_features(input_rgb, input_gray,
                                  p_pose.cx, p_pose.cy,
                                  p_windows_size[0], p_windows_size[1],
                                  p_current_scale);
        ComplexMat xf = fft2(patch_feat, p_cos_window);
        // Kernel Ridge Regression, calculate alphas (in Fourier domain)
        ComplexMat kf = gaussian_correlation(xf, xf, p_kernel_sigma, true);

        // subsequent frames, interpolate model
        p_model_xf = p_model_xf * (1. - p_interp_factor) + xf * p_interp_factor;
 //    ComplexMat alphaf = p_yf / (kf + p_lambda); //equation for fast training
 //    p_model_alphaf = p_model_alphaf * (1. - p_interp_factor) +
 //    alphaf * p_interp_factor;


        ComplexMat alphaf_num = p_yf * kf;
        ComplexMat alphaf_den = kf * (kf + p_lambda);
        p_model_alphaf_num = p_model_alphaf_num * (1. - p_interp_factor) +
           (p_yf * kf) * p_interp_factor;
        p_model_alphaf_den = p_model_alphaf_den * (1. - p_interp_factor) +
           kf * (kf + p_lambda) * p_interp_factor;
        p_model_alphaf = p_model_alphaf_num / p_model_alphaf_den;
     }
}

 // ****************************************************************************

std::vector<cv::Mat> KCF_Tracker::get_features(
    cv::Mat & input_rgb, cv::Mat & input_gray,
    int cx, int cy, int size_x, int size_y, double scale) {
    int size_x_scaled = floor(size_x*scale);
    int size_y_scaled = floor(size_y*scale);
    cv::Mat patch_gray = get_subwindow(input_gray, cx, cy,
                                       size_x_scaled, size_y_scaled);
    cv::Mat patch_rgb = get_subwindow(input_rgb, cx, cy,
                                      size_x_scaled, size_y_scaled);
    std::vector<cv::Mat> cnn_codes;
    cv::resize(patch_rgb, patch_rgb, cv::Size(p_windows_size[0],
                                              p_windows_size[1]));
    cv::Size filter_size = cv::Size(std::floor(patch_rgb.cols/p_cell_size),
                                    std::floor(patch_rgb.rows/p_cell_size));

    boost::shared_ptr<caffe::Blob<float> > blob_info (new caffe::Blob<float>);
    this->feature_extractor_->getFeatures(blob_info, cnn_codes, patch_rgb,
                                          filter_size);
    cnn_codes.clear();
    std::vector<cv::cuda::GpuMat> d_cnn_codes;
    const float *idata = blob_info->cpu_data();
    for (int i = 0; i < blob_info->channels(); i++) {
       cv::Mat im = cv::Mat::zeros(blob_info->height(),
                                   blob_info->width(), CV_32F);

       for (int y = 0; y < blob_info->height(); y++) {
          for (int x = 0; x < blob_info->width(); x++) {
             im.at<float>(y, x) = idata[
                i * blob_info->width() * blob_info->height() +
                y * blob_info->width() + x];
          }
       }
       if (filter_size.width != -1) {
          cv::resize(im, im, filter_size);
       }
       cnn_codes.push_back(im);
       d_cnn_codes.push_back(cv::cuda::GpuMat(im));
    }
     /*
       std::cout << cnn_codes[0]  << "\n\n";
       std::cout << cnn_codes[0].size()  << "\n\n\n";
     */
    return cnn_codes;
}

 /**
  * GPU
  */

// const
float* KCF_Tracker::get_featuresGPU(
    cv::Mat & input_rgb, cv::Mat & input_gray,
    int cx, int cy, int size_x, int size_y, double scale) {

    std::cout << "\33[34m GETTING FEATURES \033[0m"  << "\n";
    int size_x_scaled = floor(size_x*scale);
    int size_y_scaled = floor(size_y*scale);
    cv::Mat patch_gray = get_subwindow(input_gray, cx, cy,
                                       size_x_scaled, size_y_scaled);
    cv::Mat patch_rgb = get_subwindow(input_rgb, cx, cy,
                                      size_x_scaled, size_y_scaled);

    std::vector<cv::Mat> cnn_codes(1);  //! delete this??
    cv::resize(patch_rgb, patch_rgb, cv::Size(p_windows_size[0],
                                              p_windows_size[1]));
    cv::Size filter_size = cv::Size(std::floor(patch_rgb.cols/p_cell_size),
                                    std::floor(patch_rgb.rows/p_cell_size));
    this->window_size_ = filter_size;

    boost::shared_ptr<caffe::Blob<float> > blob_info (new caffe::Blob<float>);
    this->feature_extractor_->getFeatures(blob_info, cnn_codes, patch_rgb,
                                          filter_size);

    //! caffe ==>>> blob->cpu_data() +   blob->offset(n);
    const float *d_data = blob_info->gpu_data();

    //! interpolation
    float *d_resized_data = bilinearInterpolationGPU(
       d_data, filter_size.width, filter_size.height, blob_info->width(),
       blob_info->height(), blob_info->count(), FILTER_BATCH_);

    return d_resized_data;

    // TODO: RETURN FROM HERE

     /* // DEBUG FOR INTERPOLATION
     int o_byte = filter_size.width * filter_size.height *
                       FILTER_BATCH_ * sizeof(float);
     float *cpu_data = (float*)malloc(o_byte);
     cudaMemcpy(cpu_data, d_resized_data, o_byte, cudaMemcpyDeviceToHost);
     for (int k = 0; k < FILTER_BATCH_; k++) {
        cv::Mat im = cv::Mat::zeros(filter_size.height,
                                    filter_size.width, CV_32F);
        for (int y = 0; y < im.rows; y++) {
           for (int x = 0; x < im.cols; x++) {
              im.at<float>(y, x) = cpu_data[
                 k * im.cols * im.rows + y * im.cols + x];
           }
        }
        std::cout << "FILTER #: " << k  << "\n";
        cv::namedWindow("filter_gpu", CV_WINDOW_NORMAL);
        cv::imshow("filter_gpu", im);
        cv::namedWindow("filter_cpu", CV_WINDOW_NORMAL);
        cv::imshow("filter_cpu", debug_patch_[k]);
        cv::waitKey(0);
     }
     return;
     */ 
}


cv::Mat KCF_Tracker::trackingProcessOnGPU(float *d_features) {
   
    const int data_lenght = FILTER_SIZE_ * FILTER_BATCH_;
    float *d_cos_conv = cosineConvolutionGPU(d_features,
                                             this->d_cos_window_,
                                             data_lenght,
                                             BYTE_);
    
    cufftComplex * d_complex = cuDFT(d_cos_conv,
                                     handle_,
                                     FILTER_BATCH_, FILTER_SIZE_);

    float xf_norm_gpu =  squaredNormGPU(d_complex,
                                        FILTER_BATCH_,
                                        FILTER_SIZE_);
    float yf_norm_gpu =  squaredNormGPU(dev_model_xf_,
                                        FILTER_BATCH_,
                                        FILTER_SIZE_);

    cufftComplex *d_inv_mxf =  invComplexConjuateGPU(this->dev_model_xf_,
                                                     FILTER_BATCH_,
                                                     FILTER_SIZE_);
    cufftComplex *d_xyf = multiplyComplexGPU(d_complex,
                                             d_inv_mxf,
                                             FILTER_BATCH_ * FILTER_SIZE_);

    float *d_ifft = cuInvDFT(d_xyf, inv_handle_, FILTER_BATCH_, FILTER_SIZE_);
    float *d_xysum = invFFTSumOverFiltersGPU(d_ifft,
                                             FILTER_BATCH_,
                                             FILTER_SIZE_);

    float normalizer = 1.0f / (static_cast<float>(data_lenght));
    cuGaussianExpGPU(d_xysum, xf_norm_gpu, yf_norm_gpu,
                     p_kernel_sigma, normalizer, FILTER_SIZE_);

    cufftComplex *d_kf = convertFloatToComplexGPU(d_xysum, 1,
                                                    FILTER_SIZE_);
    cufftExecC2C(this->cufft_handle1_, d_kf, d_kf, CUFFT_FORWARD);


    cufftComplex *d_kzf = multiplyComplexGPU(this->dev_model_alphaf_,
                                             d_kf, FILTER_SIZE_);
    
    
    // TODO(ADDED TO CUFFT):
    int OUT_BYTE = FILTER_SIZE_ * sizeof(float);
    float *d_data;
    cudaMalloc(reinterpret_cast<void**>(&d_data), OUT_BYTE);
    cufftResult cufft_status = cufftExecC2R(
       this->inv_cufft_handle1_, d_kzf, d_data);
    if (cufft_status != cudaSuccess) {
       printf("cufftExecC2R failed in trackig!\n");
       exit(1);
    }
    normalizeByFactorGPU(d_data, 1, FILTER_SIZE_);
    
    // std::cout << "GPU_NORM:  " << xf_norm_gpu << " " << yf_norm_gpu  << "\n";

    float odata[FILTER_SIZE_];
    cudaMemcpy(odata, d_data, OUT_BYTE, cudaMemcpyDeviceToHost);

    cv::Mat results = cv::Mat(window_size_.height,
                              window_size_.width,
                              CV_32F, odata);

    // std::cout << results  << "\n";
    
    cudaFree(d_cos_conv);
    cudaFree(d_complex);
    cudaFree(d_features);
    cudaFree(d_inv_mxf);
    cudaFree(d_xyf);
    cudaFree(d_ifft);
    cudaFree(d_xysum);
    cudaFree(d_kf);
    cudaFree(d_kzf);
    cudaFree(d_data);
    
    return results;
}

 /**
  * END GPU
  */





cv::Mat KCF_Tracker::gaussian_shaped_labels(
   double sigma, int dim1, int dim2) {
    cv::Mat labels(dim2, dim1, CV_32FC1);
    int range_y[2] = {-dim2 / 2, dim2 - dim2 / 2};
    int range_x[2] = {-dim1 / 2, dim1 - dim1 / 2};

    double sigma_s = sigma*sigma;
    for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j) {
       float * row_ptr = labels.ptr<float>(j);
       double y_s = y*y;
       for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i) {
          row_ptr[i] = std::exp(-0.5 * (y_s + x*x) / sigma_s);
       }
    }

    // rotate so that 1 is at top-left corner (see KCF paper for explanation)
    cv::Mat rot_labels = circshift(labels, range_x[0], range_y[0]);
    // sanity check, 1 at top left corner
    assert(rot_labels.at<float>(0, 0) >= 1.f - 1e-10f);

    return rot_labels;
}

cv::Mat KCF_Tracker::circshift(
     const cv::Mat &patch, int x_rot, int y_rot) {
     cv::Mat rot_patch(patch.size(), CV_32FC1);
     cv::Mat tmp_x_rot(patch.size(), CV_32FC1);

     // circular rotate x-axis
     if (x_rot < 0) {
         // move part that does not rotate over the edge
        cv::Range orig_range(-x_rot, patch.cols);
        cv::Range rot_range(0, patch.cols - (-x_rot));
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(),
                                                             rot_range));

         // rotated part
         orig_range = cv::Range(0, -x_rot);
         rot_range = cv::Range(patch.cols - (-x_rot), patch.cols);
         patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(),
                                                              rot_range));
     } else if (x_rot > 0) {
        // move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols - x_rot);
        cv::Range rot_range(x_rot, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(),
                                                             rot_range));

         // rotated part
        orig_range = cv::Range(patch.cols - x_rot, patch.cols);
        rot_range = cv::Range(0, x_rot);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(),
                                                             rot_range));
     } else {  // zero rotation
         // move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols);
        cv::Range rot_range(0, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(),
                                                             rot_range));
     }

     // circular rotate y-axis
     if (y_rot < 0) {
        // move part that does not rotate over the edge
        cv::Range orig_range(-y_rot, patch.rows);
        cv::Range rot_range(0, patch.rows - (-y_rot));
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(
           rot_patch(rot_range, cv::Range::all()));

        // rotated part
        orig_range = cv::Range(0, -y_rot);
        rot_range = cv::Range(patch.rows - (-y_rot), patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(
           rot_patch(rot_range, cv::Range::all()));
     } else if (y_rot > 0) {
        // move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows - y_rot);
        cv::Range rot_range(y_rot, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(
           rot_patch(rot_range, cv::Range::all()));

         // rotated part
        orig_range = cv::Range(patch.rows - y_rot, patch.rows);
        rot_range = cv::Range(0, y_rot);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(
           rot_patch(rot_range, cv::Range::all()));
     } else {  // zero rotation
        // move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows);
        cv::Range rot_range(0, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(
           rot_patch(rot_range, cv::Range::all()));
     }

     return rot_patch;
}


/**
 * DFT on cuda
 */
cv::Mat KCF_Tracker::cudaFFT(const cv::Mat &input) {
   
    ROS_WARN("RUNNING: cudaFFT");
   
    FILTER_BATCH_ = 1;
    FILTER_SIZE_ = input.rows * input.cols;
    int byte = input.rows * input.step;
    float *d_input;
    cudaMalloc(reinterpret_cast<void**>(&d_input), byte);
    cudaMemcpy(d_input, reinterpret_cast<float*>(input.data),
               byte, cudaMemcpyHostToDevice);
    cufftComplex *d_output = this->cuDFT(d_input, cufft_handle1_, 1,
                                         FILTER_SIZE_);

    cufftComplex cpu_data[FILTER_SIZE_];
    cudaMemcpy(cpu_data, d_output, FILTER_SIZE_ *
               sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    
    cv::Mat output = cv::Mat(input.size(), CV_32FC2);
    for (int i = 0; i < input.rows; i++) {
       for (int j = 0; j < input.cols; j++) {
          int index = j + (i * input.cols);
          output.at<cv::Vec2f>(i, j)[0] = cpu_data[index].x;
          output.at<cv::Vec2f>(i, j)[1] = cpu_data[index].y;
       }
     }

    
    cudaFree(d_input);
    cudaFree(d_output);
    return output;
}


cv::Mat KCF_Tracker::cudaFFT2(
    float *d_input, const cv::Size in_size) {

    ROS_WARN("RUNNING: cudaFFT2");
      
   
    FILTER_BATCH_ = 1;
    FILTER_SIZE_ = in_size.width * in_size.height;
    int byte = FILTER_SIZE_ * sizeof(float);
    cufftComplex *d_output = this->cuDFT(d_input, cufft_handle1_,
                                         FILTER_BATCH_, FILTER_SIZE_);

    cufftComplex cpu_data[FILTER_SIZE_];
    cudaMemcpy(cpu_data, d_output, FILTER_SIZE_ *
               sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    cv::Mat output = cv::Mat(in_size, CV_32FC2);
    for (int i = 0; i < output.rows; i++) {
       for (int j = 0; j < output.cols; j++) {
          int index = j + (i * output.cols);
          output.at<cv::Vec2f>(i, j)[0] = cpu_data[index].x;
          output.at<cv::Vec2f>(i, j)[1] = cpu_data[index].y;
       }
    }
    // cudaFree(d_input);
    cudaFree(d_output);
    return output;
}

/**
 * END DFT on cuda
 */

ComplexMat KCF_Tracker::fft2(const cv::Mat &input) {
     cv::Mat complex_result;
     // cv::dft(input, complex_result, cv::DFT_COMPLEX_OUTPUT);
     complex_result = cudaFFT(input);
     return ComplexMat(complex_result);
}

ComplexMat KCF_Tracker::fft2(const std::vector<cv::Mat> &input,
                              const cv::Mat &cos_window) {
    int n_channels = input.size();

     ComplexMat result(input[0].rows, input[0].cols, n_channels);


     const int in_size = input[0].rows * input[0].cols;
     /*
     float *in_features = reinterpret_cast<float*>(
        std::malloc(sizeof(float) * in_size));
     */
     
     for (int i = 0; i < n_channels; ++i) {
        cv::Mat complex_result;
        // cv::dft(input[i].mul(cos_window), complex_result,
        //         cv::DFT_COMPLEX_OUTPUT);

        const int in_byte = in_size * sizeof(float);
        float *d_feat;
        cudaMalloc(reinterpret_cast<void**>(&d_feat), in_byte);
        cudaMemcpy(d_feat, reinterpret_cast<float*>(input[i].data),
                   in_byte, cudaMemcpyHostToDevice);
        float *d_cos = cosineConvolutionGPU(d_feat, this->d_cos_window_,
                                            in_size, in_byte);
        complex_result = cudaFFT2(d_cos, input[i].size());
        
        // cv::Mat in_data  = input[i].mul(cos_window);
        // complex_result = cudaFFT(in_data);

        cudaFree(d_cos);
        cudaFree(d_feat);
        
        result.set_channel(i, complex_result);
     }

     // free(in_features);
     
     return result;
}

/**
 * inverse fft
 */

cv::Mat KCF_Tracker::cudaIFFT(const cv::Mat input) {
    FILTER_BATCH_ = 1;
    FILTER_SIZE_ = input.rows * input.cols;
     int byte = FILTER_SIZE_ * sizeof(cufftComplex);

    cufftComplex *d_input;
    cudaMalloc(reinterpret_cast<void**>(&d_input), byte);

    cufftComplex cpu_data[FILTER_SIZE_];
    for (int i = 0; i < input.rows; i++) {
       for (int j = 0; j < input.cols; j++) {
          int index = j + (i * input.cols);
          cpu_data[index].x = input.at<cv::Vec2f>(i, j)[0];
          cpu_data[index].y = input.at<cv::Vec2f>(i, j)[1];
       }
    }
    cudaMemcpy(d_input, cpu_data, byte, cudaMemcpyHostToDevice);
    
    float *d_output = this->cuInvDFT(d_input, inv_cufft_handle1_,
                                     FILTER_BATCH_, FILTER_SIZE_);

    float out_data[FILTER_SIZE_];
    cudaMemcpy(out_data, d_output, FILTER_SIZE_ * sizeof(float),
               cudaMemcpyDeviceToHost);

    cv::Mat output = cv::Mat(input.size(), CV_32FC1);
    for (int i = 0; i < input.rows; i++) {
       for (int j = 0; j < input.cols; j++) {
          int index = j + (i * input.cols);
          output.at<float>(i, j) = out_data[index];
       }
    }
    
    cudaFree(d_output);
    cudaFree(d_input);

    return output;
}

cv::Mat KCF_Tracker::ifft2(const ComplexMat &inputf) {

     cv::Mat real_result;
     if (inputf.n_channels == 1) {
        // cv::dft(inputf.to_cv_mat(), real_result,
        //         cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT |
        //         cv::DFT_SCALE);
        real_result = cudaIFFT(inputf.to_cv_mat());
        
     } else {
         std::vector<cv::Mat> mat_channels = inputf.to_cv_mat_vector();
         std::vector<cv::Mat> ifft_mats(inputf.n_channels);
         for (int i = 0; i < inputf.n_channels; ++i) {
             // cv::dft(mat_channels[i], ifft_mats[i],
             //         cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT |
             //         cv::DFT_SCALE);
            ifft_mats[i] = cudaIFFT(mat_channels[i]);
         }
         cv::merge(ifft_mats, real_result);
     }
     return real_result;
}

 // hann window actually (Power-of-cosine windows)
 cv::Mat KCF_Tracker::cosine_window_function(
     int dim1, int dim2) {
     cv::Mat m1(1, dim1, CV_32FC1), m2(dim2, 1, CV_32FC1);
     double N_inv = 1./(static_cast<double>(dim1)-1.);
     for (int i = 0; i < dim1; ++i)
        m1.at<float>(i) = 0.5*(1. - std::cos(2. * CV_PI *
                                             static_cast<double>(i) * N_inv));
     N_inv = 1./ (static_cast<double>(dim2)-1.);
     for (int i = 0; i < dim2; ++i) {
        m2.at<float>(i) = 0.5*(1. - std::cos(
                                  2. * CV_PI * static_cast<double>(i) * N_inv));
     }
     cv::Mat ret = m2*m1;
     return ret;
 }

 // Returns sub-window of image input centered at [cx, cy] coordinates),
 // with size [width, height]. If any pixels are outside of the image,
 // they will replicate the values at the borders.
 cv::Mat KCF_Tracker::get_subwindow(
     const cv::Mat &input, int cx, int cy, int width, int height) {
     cv::Mat patch;

     int x1 = cx - width/2;
     int y1 = cy - height/2;
     int x2 = cx + width/2;
     int y2 = cy + height/2;

     // out of image
     if (x1 >= input.cols || y1 >= input.rows || x2 < 0 || y2 < 0) {
         patch.create(height, width, input.type());
         patch.setTo(0.f);
         return patch;
     }

     int top = 0, bottom = 0, left = 0, right = 0;

     // fit to image coordinates, set border extensions;
     if (x1 < 0) {
         left = -x1;
         x1 = 0;
     }
     if (y1 < 0) {
         top = -y1;
         y1 = 0;
     }
     if (x2 >= input.cols) {
         right = x2 - input.cols + width % 2;
         x2 = input.cols;
     } else {
         x2 += width % 2;
     }
     if (y2 >= input.rows) {
         bottom = y2 - input.rows + height % 2;
         y2 = input.rows;
     } else {
         y2 += height % 2;
     }
     if (x2 - x1 == 0 || y2 - y1 == 0) {
         patch = cv::Mat::zeros(height, width, CV_32FC1);
     } else {
        cv::copyMakeBorder(input(cv::Range(y1, y2),
                                 cv::Range(x1, x2)), patch,
                           top, bottom, left, right, cv::BORDER_REPLICATE);
     }

     // sanity check
     assert(patch.cols == width && patch.rows == height);

     return patch;

 }

ComplexMat KCF_Tracker::gaussian_correlation(
     const ComplexMat &xf, const ComplexMat &yf, double sigma,
     bool auto_correlation) {

     float xf_sqr_norm = xf.sqr_norm();

     float yf_sqr_norm = auto_correlation ? xf_sqr_norm : yf.sqr_norm();
     ComplexMat xyf = auto_correlation ? xf.sqr_mag() : xf * yf.conj();

     std::cout << "\033[33mCPU NORM IS: " << xf_sqr_norm
               << " " << yf_sqr_norm << "\033[0m\n";


     //! DEBUG
     /*
     std::cout << "\033[34m YF CONJ: \033[0m" << yf.conj().n_channels
               << " " << yf.conj().rows << "\n";

     cv::Mat check_yf = yf.conj().to_cv_mat();
     cv::Mat orign_yf = xf.to_cv_mat();
     cv::Mat xyf_cv = xyf.to_cv_mat();

     for (int j = 0; j < xyf.rows; j++) {
        for (int i = 0; i < xyf.cols; i++) {
           std::cout << check_yf.at<cv::Vec2f>(j, i)[0]  << ", ";
           std::cout << check_yf.at<cv::Vec2f>(j, i)[1]  << "\t";

           std::cout << orign_yf.at<cv::Vec2f>(j, i)[0]  << " ";
           std::cout << orign_yf.at<cv::Vec2f>(j, i)[1]  << "\t";

           std::cout << xyf_cv.at<cv::Vec2f>(j, i)[0]  << " ";
           std::cout << xyf_cv.at<cv::Vec2f>(j, i)[1]  << "\n";
        }
     }

     */
     //! END DEBUG


     // ifft2 and sum over 3rd dimension, we dont care about individual
     // channels
     cv::Mat xy_sum(xf.rows, xf.cols, CV_32FC1);
     xy_sum.setTo(0);
     cv::Mat ifft2_res = ifft2(xyf);


     std::cout << "IFFT SIZE: " << ifft2_res.size() << " "
               << ifft2_res.channels()  << "\n";
     std::cout << FILTER_SIZE_  << "\t" << xf.n_channels << " " << xf.rows << "\n";

     for (int y = 0; y < xf.rows; ++y) {
         float * row_ptr = ifft2_res.ptr<float>(y);
         float * row_ptr_sum = xy_sum.ptr<float>(y);

         for (int x = 0; x < xf.cols; ++x) {
            row_ptr_sum[x] = std::accumulate(
               (row_ptr + x*ifft2_res.channels()),
               (row_ptr + x*ifft2_res.channels() + ifft2_res.channels()), 0.f);

            //!!!!
            /*
            if (debug > 0) {
               float sum = 0.0;
               for (int k = x * ifft2_res.channels(); k <
                       x*ifft2_res.channels() + ifft2_res.channels(); k++) {
                  // std::cout << row_ptr[k] << "\n";
                  sum += row_ptr[k];
               }

               std::cout << ifft2_res.channels()  << "\n";
               std::cout << row_ptr_sum[x]  << "\t";
               // std::cout << row_ptr[x*ifft2_res.channels()]  << "\t";
               // std::cout << row_ptr[x*ifft2_res.channels() +
               // ifft2_res.channels()]  << "\n";


               std::cout << "SUM: " << sum << "\n";
               // std::cin.ignore();
               exit(-1);
            }
            //!!!!
            */
         }
     }

     float numel_xf_inv = 1.f/(xf.cols * xf.rows * xf.n_channels);

     cv::Mat tmp;
     cv::exp(- 1.f / (sigma * sigma) * cv::max(
                (xf_sqr_norm + yf_sqr_norm - 2 * xy_sum) * numel_xf_inv, 0),
             tmp);


     // std::cout << fft2(tmp)  << "\n\n";
    
    return fft2(tmp);
}

float get_response_circular(cv::Point2i &pt,
                            cv::Mat & response) {
    int x = pt.x;
    int y = pt.y;
    if (x < 0)
        x = response.cols + x;
    if (y < 0)
        y = response.rows + y;
    if (x >= response.cols)
        x = x - response.cols;
    if (y >= response.rows)
        y = y - response.rows;
    return response.at<float>(y, x);
}

cv::Point2f KCF_Tracker::sub_pixel_peak(
    cv::Point & max_loc, cv::Mat & response) {
    // find neighbourhood of max_loc (response is circular)
    // 1 2 3
    // 4   5
    // 6 7 8
    cv::Point2i p1(max_loc.x-1, max_loc.y-1),
       p2(max_loc.x, max_loc.y-1), p3(max_loc.x+1, max_loc.y-1);
    cv::Point2i p4(max_loc.x-1, max_loc.y),
       p5(max_loc.x+1, max_loc.y);
    cv::Point2i p6(max_loc.x-1, max_loc.y+1),
       p7(max_loc.x, max_loc.y+1), p8(max_loc.x+1, max_loc.y+1);

    // fit 2d quadratic function f(x, y) = a*x^2 + b*x*y + c*y^2 + d*x
    // + e*y + f
    cv::Mat A = (cv::Mat_<float>(9, 6) <<
                 p1.x*p1.x, p1.x*p1.y, p1.y*p1.y, p1.x, p1.y, 1.f,
                 p2.x*p2.x, p2.x*p2.y, p2.y*p2.y, p2.x, p2.y, 1.f,
                 p3.x*p3.x, p3.x*p3.y, p3.y*p3.y, p3.x, p3.y, 1.f,
                 p4.x*p4.x, p4.x*p4.y, p4.y*p4.y, p4.x, p4.y, 1.f,
                 p5.x*p5.x, p5.x*p5.y, p5.y*p5.y, p5.x, p5.y, 1.f,
                 p6.x*p6.x, p6.x*p6.y, p6.y*p6.y, p6.x, p6.y, 1.f,
                 p7.x*p7.x, p7.x*p7.y, p7.y*p7.y, p7.x, p7.y, 1.f,
                 p8.x*p8.x, p8.x*p8.y, p8.y*p8.y, p8.x, p8.y, 1.f,
                 max_loc.x*max_loc.x, max_loc.x*max_loc.y,
                 max_loc.y*max_loc.y, max_loc.x, max_loc.y, 1.f);
    cv::Mat fval = (cv::Mat_<float>(9, 1) <<
                    get_response_circular(p1, response),
                    get_response_circular(p2, response),
                    get_response_circular(p3, response),
                    get_response_circular(p4, response),
                    get_response_circular(p5, response),
                    get_response_circular(p6, response),
                    get_response_circular(p7, response),
                    get_response_circular(p8, response),
                    get_response_circular(max_loc, response));
    cv::Mat x;
    cv::solve(A, fval, x, cv::DECOMP_SVD);

    double a = x.at<float>(0), b = x.at<float>(1), c = x.at<float>(2),
           d = x.at<float>(3), e = x.at<float>(4);

    cv::Point2f sub_peak(max_loc.x, max_loc.y);
    if (b > 0 || b < 0) {
        sub_peak.y = ((2.f * a * e) / b - d) / (b - (4 * a * c) / b);
        sub_peak.x = (-2 * c * sub_peak.y - e) / b;
    }

    return sub_peak;
}

double KCF_Tracker::sub_grid_scale(
    std::vector<double> & responses, int index) {
    cv::Mat A, fval;
    if (index < 0 || index > static_cast<int>(p_scales.size()) - 1) {
        // interpolate from all values
        // fit 1d quadratic function f(x) = a*x^2 + b*x + c
        A.create(p_scales.size(), 3, CV_32FC1);
        fval.create(p_scales.size(), 1, CV_32FC1);
        for (size_t i = 0; i < p_scales.size(); ++i) {
            A.at<float>(i, 0) = p_scales[i] * p_scales[i];
            A.at<float>(i, 1) = p_scales[i];
            A.at<float>(i, 2) = 1;
            fval.at<float>(i) = responses[i];
        }
    } else {
       // only from neighbours
       if (index == 0 || index == static_cast<int>(p_scales.size()) - 1) {
            return p_scales[index];
       }
       A = (cv::Mat_<float>(3, 3) <<
            p_scales[index-1] * p_scales[index-1], p_scales[index-1], 1,
            p_scales[index] * p_scales[index], p_scales[index], 1,
            p_scales[index+1] * p_scales[index+1], p_scales[index+1], 1);
       fval = (cv::Mat_<float>(3, 1) << responses[index-1],
               responses[index], responses[index+1]);
    }

    cv::Mat x;
    cv::solve(A, fval, x, cv::DECOMP_SVD);
    double a = x.at<float>(0), b = x.at<float>(1);
    double scale = p_scales[index];
    if (a > 0 || a < 0)
        scale = -b / (2 * a);
    return scale;
}

//! test for cuFFT

// TODO(MOVE): to init
// cufftHandle handle_;
// cufftHandle inv_handle_;
cufftComplex* cuFFTC2Cprocess(cufftComplex *,
                              const cufftHandle,
                              const int,
                              const int);
float *cuFFTC2RProcess(cufftComplex *d_complex,
                       const cufftHandle,
                       const int,
                       const int,
                       bool = true);

cufftComplex* KCF_Tracker::cuDFT(
    float *dev_data, const cufftHandle handle,
    const int FILTER_BATCH, const int FILTER_SIZE) {

    if (FILTER_BATCH == 0 || FILTER_SIZE == 0) {
       ROS_ERROR("[cuDFT]: SIZE UNDEFINED");
       cufftComplex empty[1];
       return empty;
    }
    /*
    if (this->init_cufft_plan_) {
       cufftResult cufft_status = cufftPlan1d(
          &handle_, FILTER_SIZE_, CUFFT_C2C, FILTER_BATCH_);
       if (cufft_status != cudaSuccess) {
          ROS_ERROR("CUDAFFT PLAN [C2C] ALLOC FAILED");
       }
       cufft_status = cufftPlan1d(
          &inv_handle_, FILTER_SIZE_, CUFFT_C2R, FILTER_BATCH_);
       if (cufft_status != cudaSuccess) {
          ROS_ERROR("CUDAFFT PLAN [C2R] ALLOC FAILED");
       }
       ROS_WARN("SETUP CUFFT: %d %d", FILTER_BATCH_, FILTER_SIZE_);
       this->init_cufft_plan_ = false;
    }
    */

    cufftComplex *d_input = convertFloatToComplexGPU(
       dev_data, FILTER_BATCH, FILTER_SIZE);
    cufftComplex *d_output = cuFFTC2Cprocess(
       d_input, handle, FILTER_SIZE, FILTER_BATCH);
    cudaFree(d_input);
    return d_output;
}

float* KCF_Tracker::cuInvDFT(
    cufftComplex *d_complex, const cufftHandle handle,
    const int FILTER_BATCH, const int FILTER_SIZE) {
    // if (this->init_cufft_plan_) {
    //    ROS_FATAL("THE CUFFT PLAN IS NOT INTIALIZED");
    //    float empty[1];
    //    return empty;
    // }
    float *d_real_data = cuFFTC2RProcess(d_complex, handle,
                                         FILTER_SIZE, FILTER_BATCH, true);
    return d_real_data;
}

cufftComplex* cuFFTC2Cprocess(cufftComplex *in_data,
                              const cufftHandle handle,
                              const int FILTER_SIZE,
                              const int FILTER_BATCH) {
    const int OUT_BYTE = FILTER_SIZE * FILTER_BATCH * sizeof(cufftComplex);
    cufftResult cufft_status;

    cufftComplex *d_output;
    cudaMalloc(reinterpret_cast<void**>(&d_output), OUT_BYTE);
    cufft_status = cufftExecC2C(handle, in_data, d_output, CUFFT_FORWARD);
    // cufft_status = cufftExecC2C(handle, in_data, in_data, CUFFT_FORWARD);
    
    if (cufft_status != cudaSuccess) {
       ROS_FATAL("[cuFFTC2Cprocess]: cufftExecC2C failed!");
       std::exit(-1);  //! change to shutdown
    }
    
    return d_output;
    

    //! DEBUG
    ROS_WARN("\nWRITING TO .txt IS ENABLED");
    cufftComplex *out_data = (cufftComplex*)malloc(OUT_BYTE);
    cudaMemcpy(out_data, d_output, OUT_BYTE, cudaMemcpyDeviceToHost);
    std::ofstream outfile("cu.txt");
    for (int j = 0; j < OUT_BYTE/sizeof(cufftComplex); j++) {
       outfile << j <<  " " <<  out_data[j].x << " "<< out_data[j].y << "\n";
    }
    outfile.close();
}

float *cuFFTC2RProcess(cufftComplex *d_complex,
                       const cufftHandle handle,
                       const int FILTER_SIZE,
                       const int FILTER_BATCH, bool is_normalize) {

    if (FILTER_SIZE == 0 || FILTER_BATCH == 0) {
       printf("\033[31m ERROR: [cuFFTC2RProcess]: INPUTS = 0 \033[0m\n");
       float empty[1];
       return empty;
    }

    // cufftPlan1d(
    //    &inv_handle_, FILTER_SIZE, CUFFT_C2R, FILTER_BATCH);

    int OUT_BYTE = FILTER_SIZE * FILTER_BATCH * sizeof(float);
    float *d_data;
    cudaMalloc(reinterpret_cast<void**>(&d_data), OUT_BYTE);
    
    // cufftResult cufft_status = cufftExecC2R(
    //    inv_handle_, d_complex, (cufftReal*)d_data);
    cufftResult cufft_status = cufftExecC2R(handle, d_complex, d_data);

    
    if (cufft_status != cudaSuccess) {
       printf("cufftExecC2R failed!\n");
    }
    if (is_normalize) {
       normalizeByFactorGPU(d_data, FILTER_BATCH, FILTER_SIZE);
    }
    return d_data;
}

