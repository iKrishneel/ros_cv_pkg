//
//  Graph-Cut Segementation.cpp
//  Handheld Object Tracking
//
//  Created by Chaudhary Krishneel on 3/21/14.
//  Copyright (c) 2014 Chaudhary Krishneel. All rights reserved.
//

#include <interactive_segmentation/graph_cut_segmentation.h>

/**
 * @function to perform graphcut iteration
 * @param: img is the image to perform GC Segmentation
 *       : rect is the defined object boundary
 *       : templ_img is the output of GC segmentation
 */

cv::Mat GraphCutSegmentation::graphCutSegmentation(
    cv::Mat &image, cv::Mat &objMask, cv::Rect &rect, int iteration) {

    cv::Mat img = image(rect);
    cv::Mat o_mask = objMask(rect);
    
    /* Create object mask from the probability map*/
    cv::Mat mask = this->createMaskImage(o_mask);
    
    cv::Mat fgdModel;
    cv::Mat bgdModel;
    
    
    if (rect.width == 0 || rect.height == 0) {
       return cv::Mat();
    }
    
    // cv::Mat mask = imask(rect).clone();
    // cv::Mat img = image;  // (rect).clone();
    
    cv::grabCut(img, mask, rect, bgdModel, fgdModel,
                static_cast<int>(iteration),
                cv::GC_INIT_WITH_MASK /*cv::GC_INIT_WITH_RECT*/);
    
    /* Pixels of probable foreground is extracrted */
    cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);
    cv::Mat model = cv::Mat::zeros(img.rows, img.cols, img.type());
    img.copyTo(model, mask);
    
    /* Resize the model to the computed size */
    // cv::Rect prevRect = rect;
    // rect = this->model_resize(mask);
    
    /* Condition to avoid accumulation of small object */
    // if (rect.width < 20 && rect.height < 20) {
    //     rect = prevRect;
    //     return img;
    // }
    // if ((rect.width < (0.7f * prevRect.width) &&
    //      rect.width > (1.3f * prevRect.width)) ||
    //     (rect.height < (0.7f * prevRect.height) &&
    //      rect.height > (1.3f * prevRect.height))) {
    //     rect = prevRect;
    //     return img;
    // }
    
    // cv::imshow("mask", mask);
    cv::rectangle(model, rect, cv::Scalar(0, 255, 0), 2);
    cv::imshow("foreground", model);
    cv::imshow("object", fgdModel);
    cv::imshow("input mask", mask);
       
    // return model(rect).clone();
    return model;
}

cv::Mat GraphCutSegmentation::createMaskImage(cv::Mat &objMask) {
    cv::Mat mask = cv::Mat::zeros(objMask.rows, objMask.cols, CV_8U);
    for (int j = 0; j < objMask.rows; j++) {
        for (int i = 0; i < objMask.cols; i++) {
           if (objMask.at<float>(j, i) <= cv::GC_FGD &&
               objMask.at<float>(j, i) > cv::GC_BGD) {
              mask.at<uchar>(j, i) = cv::GC_PR_FGD;
            }
        }
    }
    return mask;
}

cv::Rect GraphCutSegmentation::model_resize(cv::Mat &src) {
    if (!src.data) {
       ROS_ERROR("No Image in GraphCutSegmentation::model_resize(.)");
       EXIT_FAILURE;
    }
    cv::Mat src_gray = src.clone();
    if (src_gray.type() != CV_8U) {
        cv::cvtColor(src, src_gray, CV_BGR2GRAY);
    }
    
    cv::Mat threshold_output;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    
    cv::threshold(src_gray, threshold_output, 0, 255, CV_THRESH_OTSU);
    cv::findContours(threshold_output, contours, hierarchy,
                     CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    
    int area = 0;
    cv::Rect rect = cv::Rect();
    
    for (int i = 0; i < contours.size(); i++) {
       cv::Rect a = cv::boundingRect(contours[i]);
        if (a.area() > area) {
            area = a.area();
            rect = a;
        }
    }
    return rect;
}
