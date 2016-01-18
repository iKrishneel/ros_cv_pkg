#! /usr/bin/env python

from interactive_segmentation.srv import *
from geometry_msgs.msg import Pose
import rospy

import numpy as numpy
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

def agglomerative_clustering(centroids, min_samples):
    cluster = 2
    datapoints = convert_pose_to_array(centroids)
    aggloc = AgglomerativeClustering(
        linkage = 'ward', n_clusters = cluster, n_components = min_samples).fit(datapoints)
    labels = aggloc.labels_
    label, indices, counts = np.unique(
        labels, return_inverse=True, return_counts=True)
    count = np.argmax(counts)
    return (labels, indices, count)

def dbscan_clustering(centroids, max_distance, min_sample):
    datapoints = convert_pose_to_array(centroids)
    db = DBSCAN(eps=max_distance, min_samples=min_sample, algorithm='auto').fit(datapoints)
    labels = db.labels_
    label, indices, counts = np.unique(
        labels, return_inverse=True, return_counts=True)
    count = 0
    for k in range(len(counts)):
        if label[k] != -1:
            if counts[k] > count:
                count = k
    if len(counts) == 1 and label[0] == -1:
        count = -1
    return (labels, indices, count)

def outlier_filtering_handler(req):
    labels, indices, elements = dbscan_clustering(req.estimated_centroids, req.max_distance, req.min_samples)
    #labels, indices, elements = agglomerative_clustering(req.estimated_centroids, req.min_samples)
    print labels, "\t", elements
    return OutlierFilteringResponse(labels, indices, elements)

def outlier_filtering_server():
    rospy.init_node('outlier_filtering_server')
    s = rospy.Service('outlier_filtering_srv',
                      OutlierFiltering,
                      outlier_filtering_handler)
    rospy.spin()

def main():
    outlier_filtering_server()
    
if __name__ == "__main__":
    main()
