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
iseg_node_ = '/interactive_segmentation/signal/target_object'
push_node_ = '/pr2_push_node/failure/signal'
merge_node_ = '/object_region_estimation/failure/signal'
bbox_node_ = '/bounding_box_handler/failure/signal' # if box is empty
grasp_node_ = '/pr2_grasp_object/failure/signal'

# flag for marked object
is_marked_object_ = False
START_SEGMENT_NODE = 1
NOT_RELEASE_OBJECT = 2  # tell PR2 not to release this object
FAILURE = -1

def nodelet_manager_signal(value, header):
    restart = Int32Stamped()
    restart.header = header
    restart.data = value
    return restart

def interactive_segmentation_callback(msg):
    global pub_signal_
    if msg.data == 1:
        rospy.loginfo("user marked object retrieved by pr2")
        hold_object = nodelet_manager_signal(NOT_RELEASE_OBJECT, msg.header)
        pub_signal_.publish(hold_object)
    
        global is_marked_object_
        is_marked_object_ = True
    elif msg.data == -1:
        rospy.logerr("segmentation node error... restaring")
        restart = nodelet_manager_signal(START_SEGMENT_NODE, msg.header)
        pub_signal_.publish(restart)

def pr2_pushing_callback(msg):
    global pub_signal_
    if msg.data == FAILURE:
        rospy.logerr("pr2 push nodelet raised error on current object")
        restart = nodelet_manager_signal(START_SEGMENT_NODE, msg.header)
        pub_signal_.publish(restart)

def object_region_estimation_callback(msg):
    global pub_signal_
    if msg.data == FAILURE:
        rospy.logerr("object merging nodelet raised error on current object")
        restart = nodelet_manager_signal(START_SEGMENT_NODE, msg.header)
        pub_signal_.publish(restart)

def grasp_object_callback(msg):
    global pub_signal_
    if is_marked_object_ and msg.data == 1:
        rospy.loginfo("THE TARGET OBJECT IS FOUND")

        # delete this
        next = nodelet_manager_signal(START_SEGMENT_NODE, msg.header)
        pub_signal_.publish(next)
        
    elif msg.data == START_SEGMENT_NODE:
        rospy.logdebug("going to segmentation node")
        next = nodelet_manager_signal(START_SEGMENT_NODE, msg.header)
        pub_signal_.publish(next)
    else:
        rospy.logfatal("SOMETHING HAS CRITICALLY GONE WRONG")
        
def subscribe():
    rospy.Subscriber(iseg_node_, Int32Stamped, interactive_segmentation_callback)
    rospy.Subscriber(push_node_, Int32Stamped, pr2_pushing_callback)
    rospy.Subscriber(merge_node_, Int32Stamped, object_region_estimation_callback)
    rospy.Subscriber(grasp_node_, Int32Stamped, grasp_object_callback)

def onInit():
    global pub_signal_
    pub_signal_ = rospy.Publisher(signal_topic_, Int32Stamped, queue_size = 10)
    subscribe()
    
    # rospy.loginfo("WAITING TO PUBLISH")
    # rospy.sleep(10.0)
    # rospy.loginfo("NODELET SETUP AND READY TO PUBLISH GO SIGNAL")
    # header = Header()
    # header.frame_id = '/base_link'
    # header.stamp = rospy.Time.now()
    # start = nodelet_manager_signal(START_SEGMENT_NODE, header)
    # pub_signal_.publish(start)
    
def main():
    rospy.init_node('interactive_segmentation_manager')    
    onInit()
    rospy.spin()

if __name__ == "__main__":
    main()
