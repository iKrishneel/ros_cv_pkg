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
from std_msgs.msg import Header

#signal topic to segmentation nodelet
signal_topic_ = '/interactive_segmentation_manager/critical/signal'
pub_signal_ = None

# incoming subscribing topics
iseg_node_ = '/interactive_segmentation/final/object'
push_node_ = '/pr2/failure/signal'
merge_node_ = '/merge/failure/signal'
grasp_node = '/grasp/failure/signal'

# flag for marked object
is_marked_object_ = False
START_SEGMENT_NODE = 1
NOT_RELEASE_OBJECT = 2

def nodelet_manager_signal(value, header):
    restart = Int32Stamped()
    restart.header = header
    restart.data = value
    return restart

def interactive_segmentation_callback(msg):
    if msg.data == 1:
        hold_object = nodelet_manager_signal(NOT_RELEASE_OBJECT, msg.header)
        pub_signal_.publish(hold_object)
    
        global is_marked_object_
        is_marked_object_ = True

def pr2_pushing_callback(msg):
    if msg.data == -1:
        restart = nodelet_manager_signal(START_SEGMENT_NODE, msg.header)
        pub_signal_.publish(restart)

def object_region_estimation_callback(msg):
    if msg.data == -1:
        restart = nodelet_manager_signal(START_SEGMENT_NODE, msg.header)
        pub_signal_.publish(restart)

def grasp_object_callback(msg):
    if is_marked_object_ and msg.data == 1:
        rospy.loginfo("THE TARGET OBJECT IS FOUND")
    elif msg.data == -1:
        next = nodelet_manager_signal(START_SEGMENT_NODE, msg.header)
        pub_signal_.publish(next)
    else:
        rospy.logfatal("SOMETHING HAS CRITICALLY GONE WRONG")
        
def subscribe():
    rospy.Subscriber(iseg_node_, Int32Stamped, interactive_segmentation_callback)
    rospy.Subscriber(push_node_, Int32Stamped, pr2_pushing_callback)
    rospy.Subscriber(merge_node_, Int32Stamped, object_region_estimation_callback)
    rospy.Subscriber(grasp_node, Int32Stamped, grasp_object_callback)

def onInit():
    pub_signal_ = rospy.Publisher(signal_topic_, Int32Stamped, queue_size = 10)
    subscribe()

    rospy.loginfo("WAITING TO PUBLISH")
    rospy.sleep(10.0)
    rospy.loginfo("NODELET SETUP AND READY TO PUBLISH GO SIGNAL")
    
    header = Header()
    header.frame_id = '/base_link'
    header.stamp = rospy.Time.now()
    start = nodelet_manager_signal(START_SEGMENT_NODE, header)
    pub_signal_.publish(start)
    
def main():
    rospy.init_node('bounding_box_manager')    
    onInit()
    rospy.spin()

if __name__ == "__main__":
    main()