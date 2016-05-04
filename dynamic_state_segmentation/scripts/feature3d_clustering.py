#! /usr/bin/env python

from dynamic_state_segmentation.srv import *
from geometry_msgs.msg import Pose
from jsk_recognition_msgs.msg import Histogram
import rospy


import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import metrics

def convert_pose_to_array(centroids):
    datapoints = []
    for i in range(len(centroids)):
        pos = [centroids[i].position.x,
               centroids[i].position.y, 
               centroids[i].position.z]
        datapoints.append(pos)
    return np.array(datapoints)

def unity_normalization(histogram):
    min_val = min(histogram)
    max_val = max(histogram)
    hist = []
    for h in histogram:
        val = (h - min_val) / (max_val - min_val)
        hist.append(val)
    return hist
        
def convert_histogram_to_array(histograms):
    hist = []
    for histogram in histograms:
        #h = unity_normalization(histogram.histogram)
        #hist.append(h)
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

def get_clusters(features, labels):
    label, indices, counts = np.unique(
        labels, return_inverse=True, return_counts=True)

    print "\n Initial", label
    
    clusters = []
    for l in label:
        cluster = []
        index = 0
        for i in indices:
            if l == i:
                cluster.append(features[index])
            index = index + 1
        if (len(cluster) > 20): # min cluster size
            clusters.append(cluster)
    return np.array(clusters)

def intra_clustering(feature, k):
    km = KMeans(n_clusters=k, init="k-means++").fit(feature)
    return (km.cluster_centers_, km.labels_)
    
def dbscan_clustering(in_features, max_distance, min_sample):
    
    max_distance = 0.05
    min_sample = 20

    features = convert_histogram_to_array(in_features)
    print "INPUT: ", features.shape
    
    #db = DBSCAN(eps=max_distance, min_samples=min_sample, algorithm='kd_tree', leaf_size=40 ).fit(features)

    #######
    bandwidth = estimate_bandwidth(features, quantile=0.1, n_samples=features.shape[0]/2)
    db = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=1).fit(features)
    
    #######
    labels = db.labels_
    

    label, indices, counts = np.unique(
        labels, return_inverse=True, return_counts=True)
    
    print label, "\t", len(db.cluster_centers_), "\t BANDWIDTH: ", bandwidth
    print "\n------\n"
    print indices
    print counts
    
    count = len(db.cluster_centers_)
    # for k in range(len(counts)):
    #     if label[k] != -1:
    #         if counts[k] > count:
    #             count = k
    # if len(counts) == 1 and label[0] == -1:
    #     count = -1
    
        
    return (labels, indices, count)

def feature3d_clustering_handler(req):
    labels, indices, elements = dbscan_clustering(req.features, req.max_distance, req.min_samples)
    #labels, indices, elements = agglomerative_clustering(req.features, req.min_samples)

    print "OUTPUT: ", labels.shape, "\t", elements
    
    #print labels, "\t", indices, "\t", elements
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
