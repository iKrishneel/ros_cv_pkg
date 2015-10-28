#!/usr/bin/env python

import roslib
roslib.load_manifest('multilayer_object_tracking')

import numpy as np
import sys
import time
import os

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
default_directory = os.path.expanduser("~/.ros/")
bag_name = ""

img = Image()
info = CameraInfo()

def get_data_time_name():
    return time.strftime("%Y-%m-%d-%H-%M-%S") + ".bag"

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
    global init_bag
    tmp_bag = rosbag.Bag(default_directory + bag_name)
    time_stamp = rospy.Time.now()
    msg_holder = []
    for topic, msg, t in tmp_bag.read_messages(topics=[sub_model_tp, sub_image_tp, sub_cam_info_tp]):
        msg_holder.append(msg)
    ros_bag_writer = rosbag.Bag(default_directory + bag_name,'w')
    for msg in msg_holder:
        msg.header.stamp = time_stamp
        ros_bag_writer.write(sub_model_tp, msg)
        # TODO: re-stamp sub-message
    points_array.header.stamp = time_stamp
    global img
    img.header.stamp = time_stamp
    global info
    info.header.stamp = time_stamp
    ros_bag_writer.write(sub_model_tp, points_array)
    ros_bag_writer.write(sub_image_tp, img)
    ros_bag_writer.write(sub_cam_info_tp, info)
    ros_bag_writer.close()

def image_cb(msg):
    global img
    img = msg
    print img.height

def camera_info_cb(msg):
    global info
    info = msg
    print info.distortion_model
        
def cloud_callback(points_array_msg):
    rosbag_write(points_array_msg)
    
    
def subscribe():
    rospy.Subscriber(sub_cam_info_tp, CameraInfo, camera_info_cb)
    rospy.Subscriber(sub_image_tp, Image, image_cb)
    #rospy.Subscriber(sub_model_tp, PointsArray, cloud_callback)

def on_init():
    subscribe()

def main():
    global bag_name
    bag_name = get_data_time_name()
    ros_bag_writer = rosbag.Bag(default_directory + bag_name,'w')
    ros_bag_writer.close()
    print "\n SETUP BAG: ", bag_name
    
    rospy.init_node('rosbag_data_recorder')
    on_init()
    #read_bag_for_test("2015-10-22-23-03-09.bag")
    rospy.spin()
    
if __name__ == "__main__":
    main()
    
