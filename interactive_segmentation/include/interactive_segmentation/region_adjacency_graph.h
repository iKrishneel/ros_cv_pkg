// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University of Tokyo

#ifndef _REGION_ADJACENCY_GRAPH_H_
#define _REGION_ADJACENCY_GRAPH_H_

#include <ros/ros.h>
#include <ros/console.h> 

#include <boost/graph/adjacency_list.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/config.hpp>

#include <pcl/features/vfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/segmentation/supervoxel_clustering.h>

#include <interactive_segmentation/constants.h>
#include <vector>

class RegionAdjacencyGraph {

#define DEBUG
   
 private:
    typedef pcl::PointXYZRGB PointT;
    struct VertexProperty {
       int v_index;
        Eigen::Vector4f v_center;
       int v_label;
       
       VertexProperty(
          int i = -1,
          Eigen::Vector4f center = Eigen::Vector4f(-1, -1, -1, -1),
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

    int comparision_points_size;
    
 public:
    RegionAdjacencyGraph();

    virtual void generateRAG(
            const std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr>,
            const std::multimap<uint32_t, uint32_t>);
    virtual void splitMergeRAG(const int = 0.0f);
    float localVoxelConvexityCriteria(
        Eigen::Vector4f, Eigen::Vector4f,
        Eigen::Vector4f, Eigen::Vector4f);
    Eigen::Vector4f cloudMeanNormal(
        const pcl::PointCloud<pcl::Normal>::Ptr, bool = true);
    /*
    virtual void getCloudClusterLabels(
       std::vector<int> &);
    */
    virtual void printGraph();
    enum {
       RAG_EDGE_WEIGHT_DISTANCE,
       RAG_EDGE_WEIGHT_CONVEX_CRITERIA
    };

    int total_label;
};
#endif  // __DEBRIS_DETECTION_REGION_ADJACENCY_GRAPH_H__
