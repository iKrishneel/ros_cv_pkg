#!/usr/bin/env python

from point_cloud_scene_decomposer.srv import *
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans

import rospy
import numpy as np

def kmeans_clustering(features):
    kmeans = KMeans(n_clusters=20, init='k-means++', n_init=20)
    kmeans.fit(features)
    kmeans_labels = kmeans.labels_
    kmeans_cluster_centers = kmeans.cluster_centers_
    kmeans_labels_unique = np.unique(kmeans_labels)
    n_clusters = len(kmeans_labels_unique)
    print("-- # of clusters: %d" % n_clusters)
    return kmeans_labels
    
def mean_shift_clustering(features):
    bandwidth = estimate_bandwidth(features, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(features)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    print("-- # of clusters: %d \n" % n_clusters)
    return labels

def cluster_voxel_srv_handler(req):
    #print 'in handler...'
    features = req.features
    stride = req.stride
    features = np.reshape(features, (-1, stride))
    # labels = mean_shift_clustering(features)
    labels = kmeans_clustering(features)
    return ClusterVoxelsResponse(labels)
    
def cluster_voxel_server():
    rospy.init_node('cluster_voxels_server')
    s = rospy.Service('cluster_voxels', ClusterVoxels, cluster_voxel_srv_handler)
    rospy.spin()
    
if __name__ == "__main__":
    cluster_voxel_server()
