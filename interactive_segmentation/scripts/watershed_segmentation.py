#!/usr/bin/env python

import rospy
import roslib
roslib.load_manifest("interactive_segmentation")

import numpy as np
import cv2
import numpy
from scipy.ndimage import label

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

sub_topic_ = '/erode_mask_image/output'
sub_topic_rgb_ = '/camera/rgb/image_rect_color'

image = None

def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None)
    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255/ncc)
    lbl[border == 255] = 255
    lbl = lbl.astype(numpy.int32)
    cv2.watershed(a, lbl)
    lbl[lbl == -1] = 0
    lbl = lbl.astype(numpy.uint8)
    return 255 - lbl

def watershed(img):

    img_gray = img
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 0, 255,
                               cv2.THRESH_OTSU)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
                               numpy.ones((3, 3), dtype=int))

    global image
    result = segment_on_dt(image, img_bin)
    
    cv2.imshow("result", result)
    cv2.imshow("bin", img_bin)
    
    result[result != 255] = 0
    result = cv2.dilate(result, None)
    image[result == 255] = (0, 0, 255)
    
    cv2.imshow("image", image)
    cv2.waitKey(3)

def cvbridge_to_cv2(img_msg, coding):
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, coding)
    except CvBridgeError as e:
        print e
    return cv_image
    
def img_callback(img_msg):
    global image
    image = cvbridge_to_cv2(img_msg, "bgr8")

def callback(img_msg):
    cv_image = cvbridge_to_cv2(img_msg, "mono8")
    if image is not None:
        watershed(cv_image)        
        
def subscribe():
    rospy.Subscriber(sub_topic_rgb_, Image, img_callback)
    rospy.Subscriber(sub_topic_, Image, callback)

def onInit():
    subscribe()

def main():
    rospy.init_node('interactive_segmentation')
    onInit()
    rospy.spin()
    

if __name__ == "__main__":
    main()
