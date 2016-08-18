#!/usr/bin/env python

import roslib
roslib.load_manifest('handheld_object_registration')
import rospy

import numpy as np
from sensor_msgs.msg import PointCloud2

pub_topic_ = '/particle_filter_update/output/template'
pub_templ_ = None

sub_annot_ = '/object_model/output/cloud'
sub_update_ = '/handheld_object_registration/output/template'

def annotation_callback(cloud_msg):
    global pub_templ_
    pub_templ_.publish(cloud_msg)

def update_callback(cloud_msg):
    global pub_templ_
    pub_templ_.publish(cloud_msg)
    
def subscribe():
    rospy.Subscriber(sub_annot_, PointCloud2, annotation_callback)
    rospy.Subscriber(sub_update_, PointCloud2, update_callback)

def onInit():
    global pub_templ_
    pub_templ_ = rospy.Publisher(pub_topic_, PointCloud2, queue_size = 10)

    subscribe()

def main():
    rospy.init_node('particle_filter_update_manager')
    onInit()
    rospy.spin()

if __name__ == "__main__":
    main()
