// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University of Tokyo

#include <interactive_segmentation/region_adjacency_graph.h>
#include <map>
#include <utility>

/**
 * constructor 
 */
RegionAdjacencyGraph::RegionAdjacencyGraph() :
    comparision_points_size(100) {
   
}

void RegionAdjacencyGraph::generateRAG(
    const std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters,
    const std::multimap<uint32_t, uint32_t> supervoxel_adjacency) {
    if (supervoxel_clusters.empty()) {
        ROS_ERROR("Empty voxel cannot generate RAG");
        return;
    }
    for (std::multimap<uint32_t, uint32_t>::const_iterator label_itr =
             supervoxel_adjacency.begin(); label_itr !=
             supervoxel_adjacency.end(); label_itr++) {
        uint32_t supervoxel_label = label_itr->first;
        VertexDescriptor centre_vertex = boost::add_vertex(
            VertexProperty(supervoxel_label, supervoxel_clusters.at(
                               supervoxel_label)->centroid_.getVector4fMap(),
                           -1), this->graph);
        for (std::multimap<uint32_t, uint32_t>::const_iterator
                 adjacent_itr = supervoxel_adjacency.equal_range(
                     supervoxel_label).first; adjacent_itr !=
                 supervoxel_adjacency.equal_range(
                     supervoxel_label).second; ++adjacent_itr) {
            uint32_t index = adjacent_itr->second;
            if (supervoxel_label != adjacent_itr->second) {
                VertexDescriptor neig_vertex = boost::add_vertex(VertexProperty(
                        adjacent_itr->second,supervoxel_clusters.at(
                            adjacent_itr->second)->centroid_.getVector4fMap(),
                        -1), this->graph);
                bool found = false;
                EdgeDescriptor e_descriptor;
                boost::tie(e_descriptor, found) = boost::edge(
                    centre_vertex, neig_vertex, this->graph);
                if (!found) {
                    boost::add_edge(centre_vertex,
                                    neig_vertex,
                                    EdgeProperty(0.0f),
                                    this->graph);
                }
            }
        }
    }
}

   

/*
void RegionAdjacencyGraph::generateRAG(
    const std::vector<pcl::PointCloud<PointT>::Ptr> &cloud_clusters,
    const std::vector<pcl::PointCloud<pcl::Normal>::Ptr>  &normal_clusters,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr centroids,
    std::vector<std::vector<int> > &neigbor_indices,
    const int edge_weight_criteria) {
    std::vector<VertexDescriptor> vertex_descriptor;
    if (cloud_clusters.size() == neigbor_indices.size()) {
       for (int j = 0; j < neigbor_indices.size(); j++) {
           VertexDescriptor r_vd; // = vertex_descriptor[j];
          for (int i = 0; i < neigbor_indices[j].size(); i++) {
             int n_index = neigbor_indices[j][i];
             VertexDescriptor vd = vertex_descriptor[n_index];
             if (r_vd != vd) {
                bool found = false;
                EdgeDescriptor e_descriptor;
                tie(e_descriptor, found) = edge(r_vd, vd, graph);
                if (!found) {
                    float distance = 0;1
                   boost::add_edge(
                      r_vd, vd, EdgeProperty(distance), this->graph);
                }
             }
          }
       }
    } else {
       ROS_WARN("Elements not same size..");
    }
}

/*
void RegionAdjacencyGraph::splitMergeRAG(
    const std::vector<pcl::PointCloud<PointT>::Ptr> &c_clusters,
    const std::vector<pcl::PointCloud<pcl::Normal>::Ptr>  &n_clusters,
    const int edge_weight_criteria,
    const int _threshold) {
    if (num_vertices(this->graph) == 0) {
       ROS_ERROR("ERROR: Cannot Merge Empty RAG ...");
       return;
    }
    std::vector<pcl::PointCloud<PointT>::Ptr> cloud_clusters;
    cloud_clusters.clear();
    cloud_clusters.insert(
       cloud_clusters.end(), c_clusters.begin(), c_clusters.end());
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normal_clusters;
    normal_clusters.clear();
    normal_clusters.insert(
       normal_clusters.end(), n_clusters.begin(), n_clusters.end());

    
    IndexMap index_map = get(boost::vertex_index, this->graph);
    EdgePropertyAccess edge_weights = get(boost::edge_weight, this->graph);
    VertexIterator i, end;
    int label = -1;
     
   for (tie(i, end) = vertices(this->graph); i != end; i++) {
        if (this->graph[*i].v_label == -1) {
           graph[*i].v_label = ++label;
        }
        AdjacencyIterator ai, a_end;
        tie(ai, a_end) = boost::adjacent_vertices(*i, this->graph);
        
        std::cout << RED << "-- ROOT: " << *i  << RED << RESET << std::endl;

        bool vertex_has_neigbor = true;
        if (ai == a_end) {
           vertex_has_neigbor = false;
           std::cout << CYAN << "NOT VERTEX " << CYAN << RESET << std::endl;
        }
        
        // while (vertex_has_neigbor) {
        for (; ai != a_end; ai++) {
           
           int neigbours_index = static_cast<int>(*ai);
              
           std::cout << BLUE << "\t Neigbour Node: " << *ai
                     << "\t " << neigbours_index
                     << BLUE << RESET  << std::endl;
           
           bool found = false;
           EdgeDescriptor e_descriptor;
           tie(e_descriptor, found) = boost::edge(
              *i, neigbours_index, this->graph);
           if (found) {
              EdgeValue edge_val = boost::get(
                 boost::edge_weight, this->graph, e_descriptor);
              float weights_ = edge_val;
              if (weights_ < _threshold) {
                 boost::remove_edge(e_descriptor, this->graph);
              } else {
                 if ((this->graph[neigbours_index].v_label == -1)) {  // ||
                    // (this->graph[neigbours_index].v_label !=
                    //  this->graph[*i].v_label)) {
                    this->graph[neigbours_index].v_label =
                       this->graph[*i].v_label;
                 }
              }
           }
        }
     }
    this->total_label = label + sizeof(char); // change to getter
#ifdef DEBUG
    // this->printGraph(this->graph);
    std::cout << MAGENTA << "\nPRINT INFO. \n --Graph Size: "
              << num_vertices(graph) << RESET <<
       std::endl << "--Total Label: " << label << "\n\n";
#endif  // DEBUG
}

void RegionAdjacencyGraph::getCloudClusterLabels(
    std::vector<int> &labelMD) {
    labelMD.clear();
    VertexIterator i, end;
    for (tie(i, end) = vertices(this->graph); i != end; ++i) {
       labelMD.push_back(static_cast<int>(this->graph[*i].v_label));
    }
}

void RegionAdjacencyGraph::printGraph(
    const Graph &_graph) {
    VertexIterator i, end;
    for (tie(i, end) = vertices(_graph); i != end; ++i) {
       AdjacencyIterator ai, a_end;
       tie(ai, a_end) = adjacent_vertices(*i, _graph);
       std::cout << *i << "\t" << _graph[*i].v_label << std::endl;
    }
}
*/