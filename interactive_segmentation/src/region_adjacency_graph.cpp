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
             supervoxel_adjacency.end(); ) {
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
            label_itr++;
        }
    }
    
    for (std::multimap<uint32_t, uint32_t>::const_iterator label_itr =
             supervoxel_adjacency.begin(); label_itr !=
             supervoxel_adjacency.end(); ) {
        uint32_t supervoxel_label = label_itr->first;
        Eigen::Vector4f c_centroid = supervoxel_clusters.at(
            supervoxel_label)->centroid_.getVector4fMap();
        Eigen::Vector4f c_normal = this->cloudMeanNormal(
            supervoxel_clusters.at(supervoxel_label)->normals_);
         std::cout << supervoxel_label << "\t";
        for (std::multimap<uint32_t, uint32_t>::const_iterator
                 adjacent_itr = supervoxel_adjacency.equal_range(
                     supervoxel_label).first; adjacent_itr !=
                 supervoxel_adjacency.equal_range(
                     supervoxel_label).second; ++adjacent_itr) {
            std::cout << adjacent_itr->second << ", ";
            if (supervoxel_label != adjacent_itr->second) {
                bool found = false;
                EdgeDescriptor e_descriptor;
                boost::tie(e_descriptor, found) = boost::edge(
                    label_itr->first, adjacent_itr->second, this->graph);
                if (!found) {
                    // Eigen::Vector4f n_centroid = supervoxel_clusters.at(
                    //     adjacent_itr->second)->centroid_.getVector4fMap();
                    // Eigen::Vector4f n_normal = this->cloudMeanNormal(
                    //     supervoxel_clusters.at(adjacent_itr->second)->normals_);
                    // float weight = this->localVoxelConvexityCriteria(
                    //     c_centroid, c_normal, n_centroid,
                    //     n_normal);
                    float weight = 0.0f;
                    boost::add_edge(supervoxel_label,
                                    adjacent_itr->second,
                                    EdgeProperty(static_cast<float>(weight)),
                                    this->graph);
                }
            }
            label_itr++;
        }
        std::cout << std::endl;
    }
    std::cout << YELLOW << "--DONE LABELING: "  << num_vertices(this->graph)
              << RESET << std::endl;
}

float RegionAdjacencyGraph::localVoxelConvexityCriteria(
    Eigen::Vector4f c_centroid, Eigen::Vector4f c_normal,
    Eigen::Vector4f n_centroid, Eigen::Vector4f n_normal) {
    c_centroid(3) = 0.0f;
    c_normal(3) = 0.0f;
    if ((n_centroid - c_centroid).dot(n_normal) > 0) {
        return 1.0f;
    } else {
        return -1.0f;
    }
}

Eigen::Vector4f RegionAdjacencyGraph::cloudMeanNormal(
    const pcl::PointCloud<pcl::Normal>::Ptr normal,
    bool isnorm) {

    if (normal->empty()) {
        return Eigen::Vector4f(0, 0, 0, 0);
    }
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    int icounter = 0;
    for (int i = 0; i < normal->size(); i++) {
        if ((!isnan(normal->points[i].normal_x)) &&
            (!isnan(normal->points[i].normal_y)) &&
            (!isnan(normal->points[i].normal_z))) {
            x += normal->points[i].normal_x;
            y += normal->points[i].normal_y;
            z += normal->points[i].normal_z;
            icounter++;
        }
    }
    Eigen::Vector4f n_mean = Eigen::Vector4f(
        x/static_cast<float>(icounter),
        y/static_cast<float>(icounter),
        z/static_cast<float>(icounter),
        0.0f);
    if (isnorm) {
        n_mean.normalize();
    }
    return n_mean;
}

void RegionAdjacencyGraph::splitMergeRAG(
    const int _threshold) {
    if (num_vertices(this->graph) == 0) {
       ROS_ERROR("ERROR: Cannot Merge Empty RAG ...");
       return;
    }
    IndexMap index_map = get(boost::vertex_index, this->graph);
    EdgePropertyAccess edge_weights = get(boost::edge_weight, this->graph);
    VertexIterator i, end;
    int label = -1;
     
    for (tie(i, end) = vertices(this->graph); i != end; i++) {
        if (this->graph[*i].v_label == -1) {
            graph
                [*i].v_label = ++label;
        }
        AdjacencyIterator ai, a_end;
        tie(ai, a_end) = boost::adjacent_vertices(*i, this->graph);
        
        std::cout << RED << "-- ROOT: " << *i  << RED << RESET << std::endl;

        bool vertex_has_neigbor = true;
        if (ai == a_end) {
            vertex_has_neigbor = false;
            std::cout << CYAN << "NOT VERTEX " << CYAN << RESET << std::endl;
        }
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
                    if ((this->graph[neigbours_index].v_label == -1)) {
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
 
/*
void RegionAdjacencyGraph::getCloudClusterLabels(
    std::vector<int> &labelMD) {
    labelMD.clear();
    VertexIterator i, end;
    for (tie(i, end) = vertices(this->graph); i != end; ++i) {
       labelMD.push_back(static_cast<int>(this->graph[*i].v_label));
    }
}
*/

void RegionAdjacencyGraph::printGraph() {
    VertexIterator i, end;
    for (tie(i, end) = vertices(this->graph); i != end; ++i) {
       AdjacencyIterator ai, a_end;
       tie(ai, a_end) = adjacent_vertices(*i, this->graph);
       std::cout << *i << "\t" << this->graph[*i].v_label  << "\t"
                 << this->graph[*i].v_index << "\t "
                 << num_vertices(this->graph)<< std::endl;
       
    }
}

