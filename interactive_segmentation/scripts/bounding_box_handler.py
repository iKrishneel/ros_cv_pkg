#!/usr/bin/env python

import roslib
roslib.load_manifest('interactive_segmentation')

import rospy
import numpy as np

from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox, Int32Stamped

boxes_topic = '/object_region_estimation_decomposer/boxes'
#boxes_topic = '/cluster_decomposer_final/boxes'
pub_box_ = None
pub_topic_ = '/bounding_box_handler/output/box'

#signal to report to manager of no box to grasp
signal_topic = '/bounding_box_handler/failure/signal'
pub_signal_ = None

def callback(msg):

    if len(msg.boxes) > 0:
        box = msg.boxes[0]  # since only one box will be published
        global pub_box_
        box.header = msg.header
        pub_box_.publish(box)
    else:  # report error to the manager
        rospy.logerr("bounding_box_handler failed. no input box")
        global pub_signal_
        signal = Int32Stamped()
        signal.header = msg.header
        signal.data = -1
        pub_signal_.publish(signal)
        
def subscribe():
    rospy.Subscriber(boxes_topic, BoundingBoxArray, callback)

def onInit():
    global pub_box_
    pub_box_ = rospy.Publisher(pub_topic_, BoundingBox, queue_size=10)

    global pub_signal_
    pub_signal_ = rospy.Publisher(signal_topic, Int32Stamped, queue_size=10)
    
    subscribe()
    
def main():
    rospy.init_node('bounding_box_handler')
    onInit()
    rospy.spin()

if __name__ == "__main__":
    main()
