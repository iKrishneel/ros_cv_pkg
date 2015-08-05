#! /usr/bin/env python

from multilayer_object_tracking.srv import *
from geometry_msgs.msg import Pose
import rospy

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics

def convert_pose_to_array(centroids):
    datapoints = []
    for center in range(len(centroids)):
        pos = [center.position.x, center.position.y, center.position.z]
        datapoints.append(pos)
    return datapoints

def run_clustering(centroids, max_distance, min_sample):
    datapoints = convert_pose_to_array(centroids)
    db = DBSCAN(eps=max_distance, min_samples=min_sample).fit(datapoints)
    """
    complete this
    """

def estimated_centroids_clustering_handler(req):
    labels = run_clustering(req.estimated_centroids,
                           req.max_distance,
                           req.min_samples)
    return EstimatedCentroidClusteringResponse(labels)

def estimated_centroid_clustering_server():
    rospy.init_node('estimated_centroids_clustering_server')
    s = rospy.Service('estimated_centroids_clustering',
                      EstimatedCentroidsClustering,
                      estimated_centroids_clustering_handler)
    rospy.spin()
    
if __name__ == "__main__":
    estimated_centroid_clustering_server()
