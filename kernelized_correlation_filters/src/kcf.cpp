
#include <kernelized_correlation_filters/kcf.h>
#include <numeric>
#include <future>
#include <thread>

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
    FILTER_SIZE_ = 256;

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
    p_yf = fft2(gaussian_shaped_labels(p_output_sigma,
                                       p_windows_size[0]/p_cell_size,
                                       p_windows_size[1]/p_cell_size));
    p_cos_window = cosine_window_function(p_yf.cols, p_yf.rows);

    // obtain a sub-window for training initial model
    std::vector<cv::Mat> path_feat = get_features(input_rgb, input_gray,
                                                  p_pose.cx, p_pose.cy,
                                                  p_windows_size[0],
                                                  p_windows_size[1]);
    p_model_xf = fft2(path_feat, p_cos_window);
    // Kernel Ridge Regression, calculate alphas (in Fourier domain)
    ComplexMat kf = gaussian_correlation(p_model_xf, p_model_xf,
                                         p_kernel_sigma, true);

    // p_model_alphaf = p_yf / (kf + p_lambda);   //equation for fast training

    p_model_alphaf_num = p_yf * kf;
    p_model_alphaf_den = kf * (kf + p_lambda);
    p_model_alphaf = p_model_alphaf_num / p_model_alphaf_den;
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

    for (size_t i = 0; i < p_scales.size(); ++i) {
       patch_feat = get_features(input_rgb, input_gray, p_pose.cx,
                                 p_pose.cy, p_windows_size[0],
                                 p_windows_size[1],
                                 p_current_scale * p_scales[i]);
       
       ComplexMat zf = fft2(patch_feat, p_cos_window);

       ROS_WARN("RUNNING CUDFF");
       std::clock_t start;
       double duration;
       start = std::clock();
           
       // this->cuDFT(patch_feat, p_cos_window);
           
       duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
       std::cout << "printf: " << duration <<'\n';

           
       ComplexMat kzf = gaussian_correlation(zf, p_model_xf,
                                             p_kernel_sigma);


           

           
       cv::Mat response = ifft2(p_model_alphaf * kzf);
            
       /* target location is at the maximum response. we must take into
          account the fact that, if the target doesn't move, the peak
          will appear at the top-left corner, not at the center (this is
          discussed in the paper). the responses wrap around cyclically. */
       double min_val, max_val;
       cv::Point2i min_loc, max_loc;
       cv::minMaxLoc(response, &min_val, &max_val, &min_loc, &max_loc);

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

    // obtain a subwindow for training at newly estimated target position
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

// ****************************************************************************

std::vector<cv::Mat> KCF_Tracker::get_features(cv::Mat & input_rgb,
                                               cv::Mat & input_gray,
                                               int cx, int cy, int size_x,
                                               int size_y, double scale) {
    std::cout << "\33[34m GETTING FEATURES \033[0m"  << "\n";
    int size_x_scaled = floor(size_x*scale);
    int size_y_scaled = floor(size_y*scale);
    cv::Mat patch_gray = get_subwindow(input_gray, cx, cy,
                                       size_x_scaled, size_y_scaled);
    cv::Mat patch_rgb = get_subwindow(input_rgb, cx, cy,
                                      size_x_scaled, size_y_scaled);


    // std::cout << patch_gray.size()  << "\t";
    // std::cout << cx << " " << cy  << "\n";
    
    std::vector<cv::Mat> cnn_codes;
    cv::resize(patch_rgb, patch_rgb, cv::Size(p_windows_size[0],
                                              p_windows_size[1]));
    cv::Size filter_size = cv::Size(std::floor(patch_rgb.cols/p_cell_size),
                                    std::floor(patch_rgb.rows/p_cell_size));
    this->feature_extractor_->getFeatures(cnn_codes, patch_rgb,
                                          filter_size);

    return cnn_codes;
}

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

ComplexMat KCF_Tracker::fft2(const cv::Mat &input) {
    cv::Mat complex_result;
    cv::dft(input, complex_result, cv::DFT_COMPLEX_OUTPUT);
    return ComplexMat(complex_result);
}

ComplexMat KCF_Tracker::fft2(const std::vector<cv::Mat> &input,
                             const cv::Mat &cos_window) {
    int n_channels = input.size();
    std::cout << "INPUT: " << input.size()  << "\n";
    ComplexMat result(input[0].rows, input[0].cols, n_channels);

    for (int i = 0; i < n_channels; ++i) {
        cv::Mat complex_result;
        cv::dft(input[i].mul(cos_window), complex_result,
                cv::DFT_COMPLEX_OUTPUT);
        /*
        if (i == 2) {
           std::ofstream outfile("cv.txt");           
           int icount = 0;
           for (int y = 0; y < complex_result.rows; y++) {
              for (int x = 0; x < complex_result.cols; x++) {
                           << complex_result.at<cv::Vec2f>(y, x)[0]  << "\t";
                 outfile << complex_result.at<cv::Vec2f>(y, x)[1]  << "\n";
              }
           }
           outfile.close();
        }
        */
        
        result.set_channel(i, complex_result);
    }
    return result;
}

cv::Mat KCF_Tracker::ifft2(const ComplexMat &inputf) {

    cv::Mat real_result;
    if (inputf.n_channels == 1) {
       cv::dft(inputf.to_cv_mat(), real_result,
               cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    } else {
        std::vector<cv::Mat> mat_channels = inputf.to_cv_mat_vector();
        std::vector<cv::Mat> ifft_mats(inputf.n_channels);
        for (int i = 0; i < inputf.n_channels; ++i) {
            cv::dft(mat_channels[i], ifft_mats[i],
                    cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
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

    // ifft2 and sum over 3rd dimension, we dont care about individual
    // channels
    cv::Mat xy_sum(xf.rows, xf.cols, CV_32FC1);
    xy_sum.setTo(0);
    cv::Mat ifft2_res = ifft2(xyf);
    for (int y = 0; y < xf.rows; ++y) {
        float * row_ptr = ifft2_res.ptr<float>(y);
        float * row_ptr_sum = xy_sum.ptr<float>(y);
        for (int x = 0; x < xf.cols; ++x) {
           row_ptr_sum[x] = std::accumulate(
              (row_ptr + x*ifft2_res.channels()),
              (row_ptr + x*ifft2_res.channels() + ifft2_res.channels()), 0.f);
        }
    }

    float numel_xf_inv = 1.f/(xf.cols * xf.rows * xf.n_channels);
    cv::Mat tmp;
    cv::exp(- 1.f / (sigma * sigma) * cv::max(
               (xf_sqr_norm + yf_sqr_norm - 2 * xy_sum) * numel_xf_inv, 0),
            tmp);

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
#include <vector>
#include <cufft.h>

// cufftHandle cufft_plan_handle_;
cufftHandle handle;
void cuFFTR2Cprocess(cufftReal *x, cufftComplex *y, size_t SIGNAL_SIZE);
   
void KCF_Tracker::cuDFT(
    const std::vector<cv::Mat> &cnn_codes,
    const cv::Mat cos_window) {
   
    if (this->init_cufft_plan_) {
       FILTER_SIZE_ = cnn_codes[2].rows * cnn_codes[2].cols;

       cufftResult cufft_status = cufftPlan1d(
          &handle, FILTER_SIZE_, CUFFT_R2C, 1);
       if (cufft_status != cudaSuccess) {
          ROS_ERROR("CUDAFFT PLAN ALLOC FAILED");
          return;
       }
       this->init_cufft_plan_ = false;
    }
    
    cv::Mat filter = cnn_codes[2].mul(cos_window);
    
    cufftReal *d_data;
    cufftReal *h_data = reinterpret_cast<cufftReal*>(filter.data);

    cufftComplex output[(FILTER_SIZE_/2) + 1];
    cuFFTR2Cprocess(h_data, output, FILTER_SIZE_);



    
    ROS_WARN("SUCCESSFULLY COMPLETED");
}

void cuFFTR2Cprocess(cufftReal *in_data,
                     cufftComplex *out_data,
                     size_t SIGNAL_SIZE) {

    const int IN_BYTE = SIGNAL_SIZE * 1 * sizeof(cufftReal);
    cufftReal *d_data;
    // cudaMalloc(reinterpret_cast<void**>(&d_data),
    //            (SIGNAL_SIZE / 2 + 1) * 1 * sizeof(cufftComplex));
    cudaMalloc(reinterpret_cast<void**>(&d_data), IN_BYTE);
    cudaMemcpy(d_data, in_data, IN_BYTE, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_data, in_data, (SIGNAL_SIZE/2 +1) *
    //            sizeof(cufftComplex), cudaMemcpyHostToDevice);


    cufftResult cufftStatus;
    // cufftHandle handle;
    // cufftStatus = cufftPlan1d(&handle, SIGNAL_SIZE, CUFFT_R2C, 1);
    // if (cufftStatus != cudaSuccess) {
    //    printf("cufftPlan1d failed!");
    // }


    const int OUT_BYTE = (SIGNAL_SIZE / 2 + 1) * sizeof(cufftComplex);
    cufftComplex *d_output;
    cudaMalloc(reinterpret_cast<void**>(&d_output), OUT_BYTE);
    
    cufftStatus = cufftExecR2C(handle,  d_data, d_output);

    if (cufftStatus != cudaSuccess) {
       printf("cufftExecR2C failed!");
    }

    cudaMemcpy(out_data, d_output, OUT_BYTE, cudaMemcpyDeviceToHost);

    // std::ofstream outfile("cu.txt");
    // for (int j = 0; j < OUT_BYTE/sizeof(cufftComplex); j++) {
    //    outfile << j <<  " " <<  out_data[j].x << " "<< out_data[j].y << "\n";
    // }
    // outfile.close();

    cufftDestroy(handle);
    cudaFree(d_data);
}
