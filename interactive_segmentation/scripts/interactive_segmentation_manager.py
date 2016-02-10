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

def bounding_box_array_callback(msg):
    rospy.loginfo("CHECKING FOR VALIDITY OF BOXES")
    for m in msg.boxes:
        box_dim = Vector3()
        box_dim = m.dimensions
        #todo: check if box is valid and then give signal if not
        print box_dim
    
        
def subscribe():
    rospy.Subscriber(topic_boxes_, BoundingBoxArray,
                     bounding_box_array_callback)

def onInit():
    global topic_boxes_
    rospy.get_param('topic_boxes', topic_boxes_)
    subscribe()

def main():
    rospy.init_node('interactive_segmentation_manager')
    onInit()
    rospy.spin()

if __name__ == "__main__":
    main()
