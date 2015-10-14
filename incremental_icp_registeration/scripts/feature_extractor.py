#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

import rospy
import roslib
roslib.load_manifest('incremental_icp_registeration')

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

image_topic = "/camera/rgb/image_rect_color"

def features_extraction(img):
    #fast = cv2.FastFeatureDetector()
    fast = cv2.SIFT()
    kp = fast.detect(img, None)
    img = cv2.drawKeypoints(img, kp, color=(0, 255, 0))

    print len(kp)
    
    cv2.imshow("image", img)
    cv2.waitKey(3)

def image_callback(image_msg):
    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")

        features_extraction(cv_image)
        
    except CvBridgeError, e:
        print e

def ros_image_bridge():
    rospy.Subscriber(image_topic, Image, image_callback)
    

def main(argv):
    ros_image_bridge()
    rospy.init_node('feature_extractor', anonymous=True)
    rospy.spin()
    
if __name__ == "__main__":
    main(sys.argv[1:])
