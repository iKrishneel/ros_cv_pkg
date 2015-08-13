#! /usr/bin/env python

from multilayer_object_tracking.srv import *
from geometry_msgs.msg import Pose
import rospy

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

def convert_pose_to_array(centroids):
    datapoints = []
    for i in range(len(centroids)):
        pos = [centroids[i].position.x,
               centroids[i].position.y, 
               centroids[i].position.z]
        datapoints.append(pos)
    return np.array(datapoints)

def agglomerative_clustering(centroids):
    cluster = 2
    datapoints = convert_pose_to_array(centroids)
    aggloc = AgglomerativeClustering(
        linkage = 'average', n_clusters = cluster).fit(datapoints)
    labels = aggloc.labels_
    label, indices, counts = np.unique(
        labels, return_inverse=True, return_counts=True)
    count = np.argmax(counts)
    return (labels, indices, count)

def dbscan_clustering(centroids, max_distance, min_sample):
    datapoints = convert_pose_to_array(centroids)
    db = DBSCAN(eps=max_distance, min_samples=min_sample, algorithm='kd_tree').fit(datapoints)
    labels = db.labels_
    label, indices, counts = np.unique(
        labels, return_inverse=True, return_counts=True)
    count = np.argmax(counts)
    return (labels, indices, count)

def estimated_centroids_clustering_handler(req):
    labels, indices, elements = dbscan_clustering(
        req.estimated_centroids, req.max_distance, req.min_samples)
    # labels, indices, elements = agglomerative_clustering(req.estimated_centroids)
    print labels, "\t", elements
    return EstimatedCentroidsClusteringResponse(labels, indices, elements)

def estimated_centroid_clustering_server():
    rospy.init_node('estimated_centroids_clustering_server')
    s = rospy.Service('estimated_centroids_clustering',
                      EstimatedCentroidsClustering,
                      estimated_centroids_clustering_handler)
    rospy.spin()

def main():
    estimated_centroid_clustering_server()
    
if __name__ == "__main__":
    main()
