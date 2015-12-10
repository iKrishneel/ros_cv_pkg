#!/usr/bin/env python

import rospy
import roslib
roslib.load_manifest('interactive_segmentation')

import numpy as np
import sys
import cv2
import math

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2

def project_pointcloud_to_image_plane(cloud, info):

    gen = pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z"))
    int_data = list(gen) 
    number_of_points = len(int_data)
    points = np.array(int_data)
    
    K = info.K
    camera_mat = np.array(K).reshape(3,3)
    
    R = info.R
    rotation_mat = np.array(R).reshape(3,3)
    rvect, jac = cv2.Rodrigues(rotation_mat)
    
    translation_mat = np.zeros((1, 3, 1), np.float32)
    translation_mat[0, 0] = info.P[3]
    translation_mat[0, 1] = info.P[7]
    translation_mat[0, 2] = info.P[11]

    D = info.D
    distortion_mat = np.array(D)
    
    image_points, jaco = cv2.projectPoints(points, rvect, translation_mat, camera_mat, distortion_mat)

    image = np.zeros((info.height, info.width, 3), np.uint8)
    image[:,0:info.width] = (255, 255, 255)
    
    for i in image_points:
        x = i[0,0]
        y = i[0,1]
        # if (not math.isnan(x) or not math.isnan(y) and
        #     (x >= 0 and x < info.width) and (y >= 0 and y < info.height)):
        image[y, x] = (0, 0, 0)
            #print i[0,0]
            
    cv2.imshow("image", image)
    cv2.waitKey(3)

camera_info_ = None   
def camera_info_callback(info_msg):
    global camera_info_
    camera_info_ = info_msg
    
def callback(cloud_msg):
    global camera_info_
    if not camera_info_ is None:
        print "PROCESSING"
        project_pointcloud_to_image_plane(cloud_msg, camera_info_)
    
def subscribe():
    sub_info = rospy.Subscriber('/camera/depth_registered/camera_info', CameraInfo, camera_info_callback)
    sub_cloud = rospy.Subscriber('/camera/depth_registered/points', PointCloud2, callback)
    
def onInit():
    subscribe()

def main():
    rospy.init_node('boundary_discontinuity_check')
    onInit()
    rospy.spin()

if __name__ == "__main__":
    main()



