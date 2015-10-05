#!/usr/bin/env python

import roslib
roslib.load_manifest('multilayer_object_tracking')

import numpy as np
import sys

import rospy
import rosbag
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge, CvBridgeError

topic_cloud = '/camera/depth_registered/points'
topic_image = '/camera/rgb/image_rect_color'
process_counter = 0
cloud_data = []
image_data = []

pub_cloud = rospy.Publisher(topic_cloud, PointCloud2, queue_size = 10)
pub_image = rospy.Publisher(topic_image, Image, queue_size = 10)

def read_rosbag(argv):
    bag = rosbag.Bag(argv[0], 'r')
    global cloud_data
    global image_data
    for topic, msg, t in bag.read_messages(topics=[topic_cloud, topic_image]):
        if topic == topic_cloud:
            cloud_data.append(msg)
        if topic == topic_image:
            image_data.append(msg)
        print topic
    bag.close()

def callback(data):
    global process_counter
    if process_counter < len(cloud_data):
        rospy.sleep(3)
        cloud = PointCloud2()
        cloud = cloud_data[process_counter]
        image = Image()
        image = image_data[process_counter]
        now = rospy.Time().now()
        cloud.header.stamp = now
        image.header.stamp = now
        pub_cloud.publish(cloud)
        pub_image.publish(image)

        process_counter += 1
    else:
        print "All Frames in the bag processed"
        sys.exit(0)
    print "Counter", process_counter

def subscribe():
    #sub_model = '/multilayer_object_tracking/output/cloud'
    sub_model = '/xtion/camera/depth_registered/points'
    rospy.Subscriber(sub_model, PointCloud2, callback)

def onInit(argv):
    read_rosbag(argv[1:])
    callback(True)
    subscribe()

def main(argv):
    rospy.init_node('rosbag_reader_publisher', argv)
    onInit(argv)
    rospy.spin()

if __name__ == "__main__":
    main(sys.argv)
