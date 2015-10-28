#!/usr/bin/env python

import numpy as np
import os
import time

import rospy
import roslib
roslib.load_manifest('multilayer_object_tracking')

import rosbag

from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from jsk_recognition_msgs.msg import PointsArray
from multilayer_object_tracking.msg import ReferenceModelBundle

write_topic = "/multilayer_object_tracking/reference_set_bundle"
default_directory = os.path.expanduser("/tmp/dataset/")

def read_rosbag_in_folder(clean_up = False):
    if os.path.exists(default_directory):
        rosbag_writer = rosbag.Bag('/tmp/dataset.bag', 'w')
        try:
            bag_array = os.listdir(default_directory)
            for bag in bag_array:
                print default_directory + bag
                
                rbag = rosbag.Bag(default_directory + bag)
                for topic, msg, t in rbag.read_messages(topics=[write_topic]):
                    rosbag_writer.write(write_topic, msg)
                rbag.close()
        finally:
            rosbag_writer.close()
    if clean_up:
        print "rm -rf ", default_directory, "[Y/n] "
        os.remove(default_directory)

if __name__ == "__main__":
    read_rosbag_in_folder(False)
                
