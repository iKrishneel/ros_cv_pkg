
#include <interactive_segmentation/saliency_map_generator.h>

SaliencyMapGenerator::SaliencyMapGenerator() {

}

bool SaliencyMapGenerator::computeSaliencyImpl(
    cv::Mat image, cv::Mat &saliencyMap) {
    if (image.channels() > 1 || image.empty()) {
       return false;
    }
    cv::Mat dst(cv::Size(image.cols, image.rows), CV_8UC1);
    calcIntensityChannel(image, dst);
    saliencyMap = cv::Mat::zeros(image.size(), CV_8UC1);
    dst.copyTo(saliencyMap);
    return true;
}

void SaliencyMapGenerator::copyImage(cv::Mat srcArg, cv::Mat dstArg) {
    srcArg.copyTo(dstArg);
}

void SaliencyMapGenerator::calcIntensityChannel(
    cv::Mat srcArg, cv::Mat dstArg) {
    if (dstArg.channels() > 1) {
        return;
    }
    const int numScales = 6;
    cv::Mat intensityScaledOn[numScales];
    cv::Mat intensityScaledOff[numScales];
    cv::Mat gray = cv::Mat::zeros(cv::Size(srcArg.cols, srcArg.rows), CV_8UC1);
    cv::Mat integralImage(cv::Size(srcArg.cols + 1, srcArg.rows + 1), CV_32FC1);
    cv::Mat intensity(cv::Size(srcArg.cols, srcArg.rows), CV_8UC1);
    cv::Mat intensityOn(cv::Size(srcArg.cols, srcArg.rows), CV_8UC1);
    cv::Mat intensityOff(cv::Size(srcArg.cols, srcArg.rows), CV_8UC1);

    int i;
    int neighborhood;
    int neighborhoods[] = {3*4, 3*4*2, 3*4*2*2, 7*4, 7*4*2, 7*4*2*2};

    for (i = 0; i < numScales; i++) {
        intensityScaledOn[i] = cv::Mat(cv::Size(
                                          srcArg.cols,
                                          srcArg.rows), CV_8UC1);
        intensityScaledOff[i] = cv::Mat(cv::Size(
                                           srcArg.cols, srcArg.rows), CV_8UC1);
    }
    if (srcArg.channels() == 3) {
        cv::cvtColor(srcArg, gray, cv::COLOR_BGR2GRAY);
    } else {
        srcArg.copyTo(gray);
    }
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0, 0);
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0, 0);
    cv::integral(gray, integralImage, CV_32F);
    for (i=0; i < numScales; i++) {
       neighborhood = neighborhoods[i];
       getIntensityScaled(integralImage, gray,
                          intensityScaledOn[i],
                          intensityScaledOff[i],
                          neighborhood);
    }
    mixScales(intensityScaledOn, intensityOn,
              intensityScaledOff,
              intensityOff,
              numScales);
    mixOnOff(intensityOn, intensityOff, intensity);
    intensity.copyTo(dstArg);
}

void SaliencyMapGenerator::getIntensityScaled(
    cv::Mat integralImage, cv::Mat gray, cv::Mat intensityScaledOn,
    cv::Mat intensityScaledOff, int neighborhood) {
    float value, meanOn, meanOff;
    cv::Point2i point;
    int x, y;
    intensityScaledOn.setTo(cv::Scalar::all(0));
    intensityScaledOff.setTo(cv::Scalar::all(0));

    for (y = 0; y < gray.rows; y++) {
       for (x = 0; x < gray.cols; x++) {
            point.x = x;
            point.y = y;
            value = getMean(integralImage,
                            point, neighborhood, gray.at<uchar>(y, x));
            meanOn = gray.at<uchar>(y, x) - value;
            meanOff = value - gray.at<uchar>(y, x);
            if (meanOn > 0)
                intensityScaledOn.at<uchar>(y, x) = (uchar)meanOn;
            else
                intensityScaledOn.at<uchar>(y, x) = 0;

            if (meanOff > 0)
                intensityScaledOff.at<uchar>(y, x) = (uchar)meanOff;
            else
                intensityScaledOff.at<uchar>(y, x) = 0;
        }
    }
}

float SaliencyMapGenerator::getMean(
    cv::Mat srcArg, cv::Point2i PixArg, int neighbourhood, int centerVal) {
    cv::Point2i P1, P2;
    float value;

    P1.x = PixArg.x - neighbourhood + 1;
    P1.y = PixArg.y - neighbourhood + 1;
    P2.x = PixArg.x + neighbourhood + 1;
    P2.y = PixArg.y + neighbourhood + 1;

    if (P1.x < 0)
        P1.x = 0;
    else if (P1.x > srcArg.cols - 1)
        P1.x = srcArg.cols - 1;
    if (P2.x < 0)
        P2.x = 0;
    else if (P2.x > srcArg.cols - 1)
        P2.x = srcArg.cols - 1;
    if (P1.y < 0)
        P1.y = 0;
    else if (P1.y > srcArg.rows - 1)
        P1.y = srcArg.rows - 1;
    if (P2.y < 0)
        P2.y = 0;
    else if (P2.y > srcArg.rows - 1)
        P2.y = srcArg.rows - 1;

    // we use the integral image to compute fast features
    value = static_cast<float> (
            (srcArg.at<float>(P2.y, P2.x)) +
            (srcArg.at<float>(P1.y, P1.x)) -
            (srcArg.at<float>(P2.y, P1.x)) -
            (srcArg.at<float>(P1.y, P2.x)));
    value = (value - centerVal)/  (( (P2.x - P1.x) * (P2.y - P1.y))-1);
    return value;
}

void SaliencyMapGenerator::mixScales(
    cv::Mat *intensityScaledOn, cv::Mat intensityOn,
    cv::Mat *intensityScaledOff, cv::Mat intensityOff, const int numScales) {
    int i = 0, x, y;
    int width = intensityScaledOn[0].cols;
    int height = intensityScaledOn[0].rows;
    short int maxValOn = 0, currValOn = 0;
    short int maxValOff = 0, currValOff = 0;
    int maxValSumOff = 0, maxValSumOn = 0;
    cv::Mat mixedValuesOn(cv::Size(width, height), CV_16UC1);
    cv::Mat mixedValuesOff(cv::Size(width, height), CV_16UC1);
    mixedValuesOn.setTo(cv::Scalar::all(0));
    mixedValuesOff.setTo(cv::Scalar::all(0));

    for (i = 0; i < numScales; i++) {
       for (y = 0; y < height; y++)
          for (x = 0; x < width; x++) {
             currValOn = intensityScaledOn[i].at<uchar>(y, x);
             if (currValOn > maxValOn)
                maxValOn = currValOn;

             currValOff = intensityScaledOff[i].at<uchar>(y, x);
             if (currValOff > maxValOff)
                maxValOff = currValOff;
             mixedValuesOn.at<unsigned short>(y, x) += currValOn;
             mixedValuesOff.at<unsigned short>(y, x) += currValOff;
         }
    }

    for (y = 0; y < height; y++)
       for (x = 0; x < width; x++) {
          currValOn = mixedValuesOn.at<unsigned short>(y, x);
          currValOff = mixedValuesOff.at<unsigned short>(y, x);
          if (currValOff > maxValSumOff)
             maxValSumOff = currValOff;
          if (currValOn > maxValSumOn)
            maxValSumOn = currValOn;
      }
    
    for (y = 0; y < height; y++)
       for (x = 0; x < width; x++) {
         intensityOn.at<uchar>(y, x) = (uchar)(255.*((float)(mixedValuesOn.at<unsigned short>(y, x) / (float)maxValSumOn)));
         intensityOff.at<uchar>(y, x) = (uchar)(255.*((float)(mixedValuesOff.at<unsigned short>(y, x) / (float)maxValSumOff)));
      }

}

void SaliencyMapGenerator::mixOnOff(
    cv::Mat intensityOn, cv::Mat intensityOff, cv::Mat intensityArg) {
    int x, y;
    int width = intensityOn.cols;
    int height = intensityOn.rows;
    int maxVal = 0;
    int currValOn, currValOff, maxValSumOff, maxValSumOn;
    cv::Mat intensity(cv::Size(width, height), CV_8UC1);
    maxValSumOff = 0;
    maxValSumOn = 0;
    for (y = 0; y < height; y++) {
       for (x = 0; x < width; x++) {
          currValOn = intensityOn.at<uchar>(y, x);
          currValOff = intensityOff.at<uchar>(y, x);
          if (currValOff > maxValSumOff) {
             maxValSumOff = currValOff;
          }
          if (currValOn > maxValSumOn) {
             maxValSumOn = currValOn;
          }
       }
    }
    if (maxValSumOn > maxValSumOff) {
        maxVal = maxValSumOn;
    } else {
        maxVal = maxValSumOff;
    }
    
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            intensity.at<uchar>(y, x) = (uchar) (
               255. * (float) (intensityOn.at<uchar>(y, x) +
                               intensityOff.at<uchar>(y, x)) / (float)maxVal);
        }
    }
    intensity.copyTo(intensityArg);
}


