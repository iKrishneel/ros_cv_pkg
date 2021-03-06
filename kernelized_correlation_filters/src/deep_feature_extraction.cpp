
#include <kernelized_correlation_filters/deep_feature_extraction.h>

FeatureExtractor::FeatureExtractor(
    const std::string m_weight, const std::string d_proto,
    const std::string mean_file,
    const std::vector<std::string> b_names, const int device_id) :
    pretrained_model_weights_(m_weight), deploy_proto_(d_proto) {
    if (this->pretrained_model_weights_.empty()) {
       ROS_FATAL("CAFFE MODEL WEIGHTS NOT FOUND!");
       return;
    }
    if (this->deploy_proto_.empty()) {
       ROS_FATAL("MODEL PROTOTXT NOT FOUND!");
       return;
    }
    
    std::cout << m_weight  << "\n";
    std::cout << d_proto  << "\n";
    std::cout << mean_file  << "\n";
    
    ROS_INFO("\033[34m -- checking blob names ...\033[0m");
    
    // this->setExtractionLayers(b_names, 1);

    ROS_INFO("\033[34m -- checking completed ...\033[0m");
    
    CHECK_GE(device_id, 0);
    caffe::Caffe::SetDevice(device_id);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);

    ROS_INFO("\033[34m -- loading up ...\033[0m");
    this->loadPreTrainedCaffeModels(mean_file);
}

void FeatureExtractor::setExtractionLayers(
    std::vector<std::string> b_names, const int min_batch) {
    for (int i = 0; i < b_names.size(); i++) {
       CHECK(feature_extractor_net_->has_blob(b_names[i]))
          << "Unknown feature blob name " << b_names[i]
          << " in the network " << this->deploy_proto_;
    }
}

bool FeatureExtractor::loadPreTrainedCaffeModels(
    const std::string mean_file) {
    this->feature_extractor_net_ =
       boost::shared_ptr<caffe::Net<float> >(
          new caffe::Net<float>(this->deploy_proto_, caffe::TEST));
    this->feature_extractor_net_->CopyTrainedLayersFrom(
       this->pretrained_model_weights_);
    
    caffe::Blob<float> *data_layer =
       this->feature_extractor_net_->input_blobs()[0];
    this->input_geometry_ = cv::Size(data_layer->width(),
                                     data_layer->height());

    this->num_channels_ = data_layer->channels();
    this->setImageNetMean(mean_file);
    return true;
}

void FeatureExtractor::getFeatures(
    boost::shared_ptr<caffe::Blob<float> > &blob_info1,
    std::vector<cv::Mat> &filters, const cv::Mat image,
    const cv::Size filter_size) {
    if (image.channels() < 3 || image.empty()) {
       ROS_FATAL("IMAGE CHANNEL IS INCORRECT");
       return;
    }
    if (this->mean_.empty()) {
       ROS_ERROR("IMAGENET MEAN NOT SET");
       return;
    }

    caffe::Blob<float> *data_layer =
       this->feature_extractor_net_->input_blobs()[0];
    data_layer->Reshape(1, this->num_channels_,
                        this->input_geometry_.height,
                        this->input_geometry_.width);
    this->feature_extractor_net_->Reshape();

    
    std::vector<cv::Mat> input_channels;
    this->wrapInputLayer(&input_channels);

    /*
    float* input_data = data_layer->mutable_cpu_data();
    const int BYTE = sizeof(float) * data_layer->height();
    std::memcpy(input_data, image.data, BYTE);
    */
    
    this->preProcessImage(image, &input_channels);
    this->feature_extractor_net_->Forward();

    /*
    caffe::Blob<float>* output_layer =
       feature_extractor_net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    std::vector<float> pred(begin, end);
    float imax = 0;
    int index = -1;
    for (int i = 0; i < pred.size(); i++) {
       if (pred[i] > imax) {
          imax = pred[i];
          index = i;
       }
    }
    std::cout << imax << " " << index  << "\n";
    return;
    */
    
    boost::shared_ptr<caffe::Blob<float> > blob_info =
       this->feature_extractor_net_->blob_by_name("conv5");

    blob_info1.reset(new caffe::Blob<float>);
    blob_info1 = blob_info;
    return;
    
    

#ifdef _DEBUG
    std::cout << "BLOB SIZE: " << blob_info->data()->size()  << "\n";
    std::cout << blob_info->height() << " " << blob_info->width()  << "\n";
    std::cout << blob_info->channels()  << "\n";
#endif

    const float *idata = blob_info->cpu_data();
    filters.clear();
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
       filters.push_back(im);
#ifdef _DEBUG_FILTERS
       cv::namedWindow("filter", CV_WINDOW_NORMAL);
       cv::resize(im, im, cv::Size(256, 256));
       cv::imshow("filter", im);
       cv::waitKey(20);
#endif
    }
    
    return;
}

void FeatureExtractor::preProcessImage(
    const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1) {
       cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4 && num_channels_ == 1) {
       cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    } else if (img.channels() == 4 && num_channels_ == 3) {
       cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    } else if (img.channels() == 1 && num_channels_ == 3) {
       cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    } else {
       sample = img;
    }
    
    cv::Mat sample_resized;
    if (sample.size() != this->input_geometry_) {
       cv::resize(sample, sample_resized, this->input_geometry_);
    } else {
       sample_resized = sample;
    }

    cv::Mat sample_float;
    if (num_channels_ == 3) {
       sample_resized.convertTo(sample_float, CV_32FC3);
    } else {
       sample_resized.convertTo(sample_float, CV_32FC1);
    }
    cv::Mat sample_normalized;
    cv::subtract(sample_float, this->mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == this->feature_extractor_net_->input_blobs()[0]->cpu_data())
       << "Input channels are not wrapping the input layer of the network.";
}

void FeatureExtractor::wrapInputLayer(
    std::vector<cv::Mat>* input_channels) {
    caffe::Blob<float>* input_layer =
       this->feature_extractor_net_->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
       cv::Mat channel(height, width, CV_32FC1, input_data);
       input_channels->push_back(channel);
       input_data += width * height;
    }
}


bool FeatureExtractor::setImageNetMean(
    const std::string mean_file) {
    if (mean_file.empty()) {
       ROS_FATAL("MEAN FILE NOT FOUND");
       return false;
    }

    std::cout << mean_file  << "\n";
    
    caffe::BlobProto blob_proto;
    caffe::ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    
    /* Convert from BlobProto to Blob<float> */
    caffe::Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), this->num_channels_)
       << "Number of channels of mean file doesn't match input layer.";

    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
       /* Extract an individual channel. */
       cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
       channels.push_back(channel);
       data += mean_blob.height() * mean_blob.width();
    }

    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
          * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    this->mean_ = cv::Mat(this->input_geometry_, mean.type(), channel_mean);

}
