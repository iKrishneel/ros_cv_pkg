#ifndef KCF_HEADER_6565467831231
#define KCF_HEADER_6565467831231

#include <kernelized_correlation_filters/deep_feature_extraction.h>
#include <kernelized_correlation_filters/fast_maths_kernel.h>
#include <kernelized_correlation_filters/cosine_convolution_kernel.h>
#include <kernelized_correlation_filters/gaussian_correlation_kernel.h>
#include <kernelized_correlation_filters/bilinear_interpolation_kernel.h>
#include <kernelized_correlation_filters/discrete_fourier_transform_kernel.h>
#include <kernelized_correlation_filters/complexmat.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaarithm.hpp>

struct BBox_c {
    double cx, cy, w, h;
    inline void scale(double factor) {
        cx *= factor;
        cy *= factor;
        w  *= factor;
        h  *= factor;
    }
    inline cv::Rect get_rect() {
        return cv::Rect(cx-w/2., cy-h/2., w, h);
    }
};

class KCF_Tracker {
   
 private:
    BBox_c p_pose;
    bool p_resize_image;

    double p_padding;
    double p_output_sigma_factor;
    double p_output_sigma;
    double p_kernel_sigma;
    double p_lambda;
    double p_interp_factor;
    int p_cell_size;
    cv::Mat p_cos_window;
    int p_num_scales;
    double p_scale_step;
    double p_current_scale;

    int p_windows_size[2];
    double p_min_max_scale[2];
    std::vector<double> p_scales;

    // model
    ComplexMat p_yf;
    ComplexMat p_model_alphaf;
    ComplexMat p_model_alphaf_num;
    ComplexMat p_model_alphaf_den;
    ComplexMat p_model_xf;

    //! gpu model
    cufftComplex *dev_p_yf_;
    cufftComplex *dev_model_alphaf_;
    cufftComplex *dev_model_alphaf_num_;
    cufftComplex *dev_model_alphaf_den_;
    cufftComplex *dev_model_xf_;

   
    // helping functions
    cv::Mat get_subwindow(const cv::Mat & input, int cx, int cy,
                          int size_x, int size_y);
    cv::Mat gaussian_shaped_labels(double sigma, int dim1, int dim2);
    ComplexMat gaussian_correlation(const ComplexMat & xf,
                                    const ComplexMat & yf, double sigma,
                                    bool auto_correlation = false);
    cv::Mat circshift(const cv::Mat & patch, int x_rot, int y_rot);
    cv::Mat cosine_window_function(int dim1, int dim2);
    ComplexMat fft2(const cv::Mat & input);
    ComplexMat fft2(const std::vector<cv::Mat> & input,
                    const cv::Mat & cos_window);
    cv::Mat ifft2(const ComplexMat & inputf);
    std::vector<cv::Mat> get_features(cv::Mat &, cv::Mat &, int,
                                      int, int, int, double = 1.0f);
    cv::Point2f sub_pixel_peak(cv::Point &, cv::Mat &);
    double sub_grid_scale(std::vector<double> &,
                          int index = -1);

 protected:
    FeatureExtractor *feature_extractor_;

    int FILTER_SIZE_;  //! size of cnn codes
    int FILTER_BATCH_;  //! batch size
    bool init_cufft_plan_;
    float *d_cos_window_;
    int BYTE_;
    cv::Size window_size_;

    //! handle with batch = 1
    cufftHandle cufft_handle1_;
    cufftHandle inv_cufft_handle1_;

    //! main handle
    cufftHandle handle_;
    cufftHandle inv_handle_;


    //! DEGUB
    ComplexMat result_;
   
 public:
    bool m_use_scale;
    bool m_use_color;
    bool m_use_subpixel_localization;
    bool m_use_subgrid_scale;
    bool m_use_multithreading;
    bool m_use_cnfeat;
    bool is_cnn_set_;

    KCF_Tracker(double, double, double, double, double, int cell_size);
    KCF_Tracker();
   
    // Init/re-init methods
    void init(cv::Mat &, const cv::Rect &);
    void setTrackerPose(BBox_c &, cv::Mat &);
    void updateTrackerPosition(BBox_c &);

    // frame-to-frame object tracking
    void track(cv::Mat &);
    BBox_c getBBox();

    void setCaffeInfo(const std::string, const std::string,
                      const std::string,
                      std::vector<std::string> &,
                      const int);


    //! added methods
    cufftComplex* cuDFT(float *, const cufftHandle,
                        const int, const int);
    float* cuInvDFT(cufftComplex *, const cufftHandle,
                    const int, const int);

    float* get_featuresGPU(cv::Mat &, cv::Mat &,
                           int, int, int, int, double);
    cv::Mat trackingProcessOnGPU(float *);

    // DEGUB
    cv::Mat cudaFFT(const cv::Mat &);
    ComplexMat cudaFFT2(float *, const cv::Size, const int);
    cv::Mat cudaIFFT(const cv::Mat);
    cv::Mat cudaIFFT2(cufftComplex*, const cv::Size);
   
};

#endif //KCF_HEADER_6565467831231
