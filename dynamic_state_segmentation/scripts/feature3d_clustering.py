#! /usr/bin/env python

from dynamic_state_segmentation.srv import *
from geometry_msgs.msg import Pose
from jsk_recognition_msgs.msg import Histogram
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

def convert_histogram_to_array(histograms):
    hist = []
    for histogram in histograms:
        hist.append(histogram.histogram)
    return np.array(hist)

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



def dbscan_clustering(in_features, max_distance, min_sample):
    
    max_distance = 0.04
    min_sample = 50
    # print in_features[0].histogram
    # print in_features[1].histogram
    # print "\n"
    
    features = convert_histogram_to_array(in_features)
    print "INPUT: ", features.shape
    
    db = DBSCAN(eps=max_distance, min_samples=min_sample, algorithm='auto').fit(features)
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

def feature3d_clustering_handler(req):
    labels, indices, elements = dbscan_clustering(req.features, req.max_distance, req.min_samples)
    #labels, indices, elements = agglomerative_clustering(req.features, req.min_samples)

    print "OUTPUT: ", labels.shape
    
    print labels, "\t", indices, "\t", elements
    return Feature3DClusteringResponse(labels, indices, elements)

def feature3d_clustering_server():
    rospy.init_node('feature3d_clustering_server')
    s = rospy.Service('feature3d_clustering_srv',
                      Feature3DClustering,
                      feature3d_clustering_handler)
    rospy.spin()

def main():
    feature3d_clustering_server()
    
if __name__ == "__main__":
    main()
