#!/usr/bin/env python

import roslib
roslib.load_manifest('multilayer_object_tracking')

import numpy as np
import sys
import time
import os
import shutil

import rospy
import rosbag

from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from jsk_recognition_msgs.msg import PointsArray
from multilayer_object_tracking.msg import ReferenceModelBundle

sub_model_tp = '/multilayer_object_tracking/output/model_surfels_set'
sub_image_tp = '/camera/rgb/image_rect_color'
sub_cam_info_tp = '/camera/depth_registered/camera_info'

init_bag = True
default_directory = os.path.expanduser("/tmp/")
bag_name = None
class_label_ = None

img = Image()
info = CameraInfo()

def get_data_time_name():
    return time.strftime("%Y-%m-%d-%H-%M-%S") + ".bag"

def setup_folder():
    global default_directory
    folder_name = 'dataset'
    if os.path.exists(default_directory + folder_name):
        os.rename(default_directory + folder_name, default_directory +
                  str(time.strftime("%Y-%m-%d-%H-%M-%S")))
        #os.remove(default_directory + folder_name)
    os.makedirs(default_directory + folder_name)
    default_directory += (folder_name + '/')

def read_bag_for_test(bag_to_read):
    tmp_bag = rosbag.Bag(default_directory + bag_to_read)
    pub = rospy.Publisher('test_topic', PointCloud2, queue_size=10)
    for topic, msg, t in tmp_bag.read_messages(topics=[sub_model_tp]):
        cloud = PointCloud2()
        cloud = msg.cloud_list[len(msg.cloud_list)-1]
        cloud.header.stamp = rospy.Time.now()
        print cloud.header.frame_id
        pub.publish(cloud)
        print "Published ONE"
        rospy.sleep(3)
    print "DONE:"
        
def rosbag_write(points_array):

    time_stamp = rospy.Time.now()

    try:
        os.remove(default_directory + str(get_data_time_name()))
    except OSError:
        pass
        
    ros_bag_writer = rosbag.Bag(default_directory + str(get_data_time_name()),'w')
    # points_array.header.stamp = time_stamp
    global img
    img.header.stamp = points_array.header.stamp
    global info
    info.header.stamp = points_array.header.stamp
    
    bundle = ReferenceModelBundle()
    bundle.cloud_bundle = points_array
    bundle.image_bundle = img
    bundle.cam_info = info
    bundle.label = class_label_ # label of the class
    bundle.header = points_array.header
    
    # ros_bag_writer.write(sub_model_tp, points_array)
    # ros_bag_writer.write(sub_image_tp, img)
    # ros_bag_writer.write(sub_cam_info_tp, info)
    write_topic = "/multilayer_object_tracking/reference_set_bundle"
    ros_bag_writer.write(write_topic, bundle)
    ros_bag_writer.close()

def image_cb(msg):
    global img
    img = msg

def camera_info_cb(msg):
    global info
    info = msg
        
def cloud_callback(points_array_msg):
    rosbag_write(points_array_msg)
    
def subscribe():
    rospy.Subscriber(sub_cam_info_tp, CameraInfo, camera_info_cb)
    rospy.Subscriber(sub_image_tp, Image, image_cb)
    rospy.Subscriber(sub_model_tp, PointsArray, cloud_callback)

def on_init():
    subscribe()

def main():
    global class_label_
    class_label_ = rospy.get_param('/rosbag_data_recorder/class_label')
    print "LABEL: ", class_label_
    if class_label_ is None or class_label_ == 'none':
        rospy.signal_shutdown("THE CLASS LABEL IS NOT SET")
    
    setup_folder()
    print "\n SETUP BAG: ", default_directory
    
    rospy.init_node('rosbag_data_recorder')
    on_init()
    rospy.spin()
    
if __name__ == "__main__":
    main()
    
