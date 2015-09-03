#!/usr/bin/env python

import numpy as np
import sys
import time

import rosbag
import rospy

import message_filters
from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image

from geometry_msgs.msg import PoseStamped
from jsk_recognition_msgs.msg import BoundingBoxArray

import roslib
roslib.load_manifest('multilayer_object_tracking')

topic_cloud = '/camera/depth_registered/points'
topic_image = '/camera/rgb/image_rect_color'
topic_model = '/multilayer_object_tracking/ouput/cloud'
topic_templ = '/multilayer_object_tracking/output/template'
topic_bbox = '/bounding_box_estimation/output/bounding_box'
publisher_counter = 0



class RosBagWriteTrackingInfo:

    def __init__(self):
        # self.name = self.get_default_name()
        self.name = "default.bag"
        self.bag_file = rosbag.Bag(self.name, 'w')
        self.stop_counter = 100
        
        self.cloud_data = PointCloud2()
        self.image_data_array = []
        
        self.sub_cloud = rospy.Subscriber(topic_cloud, PointCloud2, self.cloub_callback)
        #self.sub_model = message_filters.Subscriber(topic_model, PointCloud2, self.model_callback)
        #self.sub_template = message_filters.Subscriber(topic_templ, PointCloud2, self.template_callback)
        #self.sub_bbox = message_filters.Subscriber(topic_bbox, BoundingBoxArray, self.bbox_callback)
        self.sub_image = rospy.Subscriber(topic_image, Image, self.image_callback)
        

    def get_default_name(self):
        timestr = time.strftime("%Y-%m-%d-%H-%M-%S") + ".bag"
        return timestr

    def cloub_callback(self, cloud):
        print "Cloud Called"
        self.cloud_data = cloud
        self.cloud_data.header.frame_id = "camera_link"
                    
    def image_callback(self, image):
        print "Image Called"
        image_data = image
        if self.stop_counter > 0:
            self.bag_file.write(topic_image, image_data)
            self.bag_file.write(topic_cloud, self.cloud_data)

        if self.stop_counter == 0:
            self.write_to_bag()
        self.stop_counter = self.stop_counter - 1
        print self.stop_counter

    def model_callback(self, cloud):
        print "Model Called"

    def template_callback(self, cloud):
        print "template Called"

    def bbox_callback(self, bbox):
        print "Bbox Called"

    def write_to_bag(self):
        self.bag_file.close()
        print "Wrote To Bag"
        
                
            
        
def main(bag_path):

    rospy.init_node('rosbag_publisher', anonymous=True)
    rosbag_proc = RosBagWriteTrackingInfo()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting Down Node"
    

if __name__ == "__main__":
    main(sys.argv)
