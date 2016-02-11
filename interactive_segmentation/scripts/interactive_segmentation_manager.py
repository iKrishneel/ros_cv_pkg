#!/usr/bin/env python

import roslib
roslib.load_manifest('interactive_segmentation')

import rospy
import numpy as np
import sys

from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from jsk_recognition_msgs.msg import Int32Stamped, BoolStamped
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import Pose, Vector3

topic_boxes_ = '/cluster_decomposer_final/boxes'
boxes_topic_ = '/interactive_segmentation_manager/output/boxes'
pub_boxes_ = None

signal_topic_ = '/interactive_segmentation_manager/output/signal'
pub_signal_ = None

min_threshold_x_ = 0.04
min_threshold_y_ = 0.04
min_threshold_z_ = 0.04

def bounding_box_array_callback(msg):
    rospy.loginfo("CHECKING FOR VALIDITY OF BOXES")
    valid_boxes = BoundingBoxArray()
    for m in msg.boxes:
        box_dim = Vector3()
        box_dim = m.dimensions
        if (box_dim.x > min_threshold_x_) and \
           (box_dim.y > min_threshold_y_) and \
           (box_dim.z > min_threshold_z_):
            valid_boxes.boxes.append(m)
    if (len(valid_boxes.boxes) == 0):
        int_stamp = Int32Stamped()
        int_stamp.header = msg.header
        int_stamp.data = -1
        pub_signal_.publish(int_stamp)
    else:
        valid_boxes.header = msg.header
        pub_boxes_.publish(valid_boxes)
        
def subscribe():
    rospy.Subscriber(topic_boxes_, BoundingBoxArray,
                     bounding_box_array_callback)

def onInit():
    global topic_boxes_
    rospy.get_param('topic_boxes', topic_boxes_)

    print  topic_boxes_
    
    global pub_boxes_
    pub_boxes_ = rospy.Publisher(boxes_topic_, BoundingBoxArray)
    global pub_signal_
    pub_signal_ = rospy.Publisher(signal_topic_, Int32Stamped)
    subscribe()

def main():
    rospy.init_node('interactive_segmentation_manager')
    onInit()
    rospy.spin()

if __name__ == "__main__":
    main()
