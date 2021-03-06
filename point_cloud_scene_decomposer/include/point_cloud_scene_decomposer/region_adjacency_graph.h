// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University of Tokyo

#ifndef _REGION_ADJACENCY_GRAPH_H_
#define _REGION_ADJACENCY_GRAPH_H_

#include <point_cloud_scene_decomposer/constants.h>

// boost header directives
#include <boost/graph/adjacency_list.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/config.hpp>

// PCL Feature
#include <pcl/features/vfh.h>
#include <pcl/features/fpfh.h>

#include <vector>

class RegionAdjacencyGraph {

#define DEBUG
   
 private:
    struct VertexProperty {
       int v_index;
       pcl::PointXYZ v_center;
       int v_label;
       
       VertexProperty(
          int i = -1,
          pcl::PointXYZ center = pcl::PointXYZ(-1, -1, -1),
          int label = -1) :
          v_index(i), v_center(center), v_label(label) {}
    };
    typedef boost::property<boost::edge_weight_t, float> EdgeProperty;
    typedef typename boost::adjacency_list<boost::vecS,
                                           boost::vecS,
                                           boost::undirectedS,
                                           VertexProperty,
                                           EdgeProperty> Graph;
    typedef typename boost::graph_traits<
       Graph>::adjacency_iterator AdjacencyIterator;
    typedef typename boost::property_map<
      Graph, boost::vertex_index_t>::type IndexMap;
    typedef typename boost::graph_traits<
       Graph>::edge_descriptor EdgeDescriptor;
    typedef typename boost::property_map<
       Graph, boost::edge_weight_t>::type EdgePropertyAccess;
    typedef typename boost::property_traits<boost::property_map<
      Graph, boost::edge_weight_t>::const_type>::value_type EdgeValue;
    typedef typename boost::graph_traits<
       Graph>::vertex_iterator VertexIterator;
    typedef typename boost::graph_traits<
       Graph>::vertex_descriptor VertexDescriptor;
   
    Graph graph;

    void sampleRandomPointsFromCloudCluster(
      pcl::PointCloud<PointT>::Ptr,
      pcl::PointCloud<pcl::Normal>::Ptr,
      std::vector<Eigen::Vector3f> &,
      std::vector<Eigen::Vector3f> &,
      int = 3);
    template<typename T>
    T convexityCriterion(
       const Eigen::Vector3f &,
       const Eigen::Vector3f &,
       const Eigen::Vector3f &,
       const Eigen::Vector3f &);

    template<typename T>
     T getCloudClusterWeightFunction(
        const std::vector<std::vector<Eigen::Vector3f> > &,
        const std::vector<std::vector<Eigen::Vector3f> > &);

    float getVectorAngle(
      const Eigen::Vector3f &,
      const Eigen::Vector3f &,
      bool = true);
    int getCommonNeigbour(
        const std::vector<int> &,
        const std::vector<int> &);

    void computeCloudClusterRPYHistogram(
        const pcl::PointCloud<PointT>::Ptr,
        const pcl::PointCloud<pcl::Normal>::Ptr,
        cv::Mat &);
    void computeColorHistogram(
       const pcl::PointCloud<PointT>::Ptr,
       cv::Mat &,
       const int = 90,
       const int = 128,
       bool = true);

    float getEdgeWeight(
        const cv::Mat &, const cv::Mat &,
        const cv::Mat &, const cv::Mat &);

    int comparision_points_size;
    
 public:
    RegionAdjacencyGraph();

    virtual void generateRAG(
       const std::vector<pcl::PointCloud<PointT>::Ptr> &,
       const std::vector<pcl::PointCloud<pcl::Normal>::Ptr> &,
       const pcl::PointCloud<pcl::PointXYZ>::Ptr,
       std::vector<std::vector<int> > &,
       const int = RAG_EDGE_WEIGHT_DISTANCE);
    virtual void splitMergeRAG(
        const std::vector<pcl::PointCloud<PointT>::Ptr> &,
        const std::vector<pcl::PointCloud<pcl::Normal>::Ptr> &,
        const int,
        const int = 0.0f);
    virtual void getCloudClusterLabels(
       std::vector<int> &);
    virtual void printGraph(
       const Graph &);

    virtual void mergePointCloud(
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<pcl::Normal>::Ptr,
       const pcl::PointCloud<pcl::Normal>::Ptr,
       pcl::PointCloud<PointT>::Ptr,
       pcl::PointCloud<pcl::Normal>::Ptr);
   
    enum {
       RAG_EDGE_WEIGHT_DISTANCE,
       RAG_EDGE_WEIGHT_CONVEX_CRITERIA
    };

    int total_label;
};
#endif  // __DEBRIS_DETECTION_REGION_ADJACENCY_GRAPH_H__
