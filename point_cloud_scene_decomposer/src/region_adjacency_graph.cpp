// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University of Tokyo

#include <point_cloud_scene_decomposer/region_adjacency_graph.h>
#include <map>
#include <utility>

/**
 * constructor 
 */
RegionAdjacencyGraph::RegionAdjacencyGraph() {
   
}

/**
 * create the RAG tree from the centroid and its neigboring distances
 */
void RegionAdjacencyGraph::generateRAG(
    const std::vector<pcl::PointCloud<PointT>::Ptr> &cloud_clusters,
    const std::vector<pcl::PointCloud<pcl::Normal>::Ptr>  &normal_clusters,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr centroids,
    std::vector<std::vector<int> > &neigbor_indices,
    const int edge_weight_criteria) {
    if (cloud_clusters.empty() || normal_clusters.empty() ||
        centroids->empty() || neigbor_indices.empty()) {
       ROS_ERROR("ERROR: Cannot Generate RAG of empty data...");
       return;
    }
    const int comparision_points_size = 100;
    if (cloud_clusters.size() == neigbor_indices.size()) {
       std::vector<VertexDescriptor> vertex_descriptor;
       for (int j = 0; j < centroids->size(); j++) {
          VertexDescriptor v_des = add_vertex(
             VertexProperty(j, centroids->points[j], -1), this->graph);
          vertex_descriptor.push_back(v_des);
       }
       for (int j = 0; j < neigbor_indices.size(); j++) {
          VertexDescriptor r_vd = vertex_descriptor[j];
          // TODO(ADD WEIGHT COMPUTING FUNCTION)
          std::vector<Eigen::Vector3f> center_point;
          std::vector<Eigen::Vector3f> center_normal;
           cv::Mat r_histogram;
           if (edge_weight_criteria == RAG_EDGE_WEIGHT_CONVEX_CRITERIA) {
             this->sampleRandomPointsFromCloudCluster(
                cloud_clusters[j],
                normal_clusters[j],
                center_point,
                center_normal,
                comparision_points_size);
             } else if (edge_weight_criteria == RAG_EDGE_WEIGHT_DISTANCE) {
             this->computeCloudClusterRPYHistogram(
                cloud_clusters[j],
                normal_clusters[j],
                r_histogram);
          } else {
              ROS_ERROR("Incorrect Measurement type ==> Setting to Default");
          }
          for (int i = 0; i < neigbor_indices[j].size(); i++) {
             int n_index = neigbor_indices[j][i];
             VertexDescriptor vd = vertex_descriptor[n_index];
              // TODO(ADD WEIGHT COMPUTING FUNCTION)
             float distance = 0.0f;
             if (edge_weight_criteria == RAG_EDGE_WEIGHT_CONVEX_CRITERIA) {
                std::vector<Eigen::Vector3f> n1_point;
                std::vector<Eigen::Vector3f> n1_normal;
                this->sampleRandomPointsFromCloudCluster(
                   cloud_clusters[n_index],
                   normal_clusters[n_index],
                   n1_point,
                   n1_normal,
                   comparision_points_size);
                // Common Neigbour
                int commonIndex = -1;
                   this->getCommonNeigbour(
                   neigbor_indices[j],
                   neigbor_indices[n_index]);
                if (commonIndex == -1) {
                   // distance = this->getCloudClusterWeightFunction<float>(
                   //    center_point, n1_point, center_normal, n1_normal);
                } else {
                   std::vector<Eigen::Vector3f> n2_point;
                   std::vector<Eigen::Vector3f> n2_normal;
                   this->sampleRandomPointsFromCloudCluster(
                      cloud_clusters[commonIndex],
                      normal_clusters[commonIndex],
                      n2_point,
                      n2_normal,
                      comparision_points_size);
                   std::vector<std::vector<Eigen::Vector3f> > _points;
                   std::vector<std::vector<Eigen::Vector3f> > _normals;
                   _points.push_back(center_point);
                   _points.push_back(n1_point);
                   // _points.push_back(n2_point);
                   _normals.push_back(center_normal);
                   _normals.push_back(n1_normal);
                   // _normals.push_back(n2_normal);
                   distance = this->getCloudClusterWeightFunction<float>(
                      _points, _normals);
                }
             } else if (edge_weight_criteria == RAG_EDGE_WEIGHT_DISTANCE) {
                cv::Mat n_histogram;
                this->computeCloudClusterRPYHistogram(
                   cloud_clusters[n_index],
                   normal_clusters[n_index],
                   n_histogram);
                distance = static_cast<float>(
                   cv::compareHist(
                      r_histogram, n_histogram, CV_COMP_CORREL));
                
             } else {
                distance = 0.0f;
             }
             if (r_vd != vd) {
                bool found = false;
                EdgeDescriptor e_descriptor;
                tie(e_descriptor, found) = edge(r_vd, vd, graph);
                if (!found) {
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

/**
 * return the weight of the point cloud. The arguments are array with
 * indices arranged as:
 * {0,1,2} ==> {i,j,c} 
 */
template<typename T>
T RegionAdjacencyGraph::getCloudClusterWeightFunction(
    const std::vector<std::vector<Eigen::Vector3f> > &_points,
    const std::vector<std::vector<Eigen::Vector3f> > &_normal) {
#define ANGLE_THRESHOLD (10)
   
    if (_points.size() == 2 && _points.size() == _normal.size()) {
       T weights_ = -1.0f;
       int concave_ = 0;
       int convex_ = 0;
       for (int i = 0; i < _points[0].size(); i++) {
          T convexC_ij = this->convexityCriterion<T>(
             _points[0][i], _points[1][i], _normal[0][i], _normal[1][i]);
          float angle_ = getVectorAngle(_normal[0][i], _normal[1][i]);
          if (convexC_ij < 0.0f && angle_ < ANGLE_THRESHOLD) {
             convexC_ij = abs(convexC_ij);
          }
          if (convexC_ij > 0.0) {
             convex_++;
          }
          if (convexC_ij <= 0.0 || isnan(convexC_ij)) {
             concave_++;
          }
          /*
          if (convexC_ij > weights_) {
             weights_ = convexC_ij;
             }*/
       }
       if (concave_ < convex_ + 5) {
          weights_ = 1.0f;
       }
       return weights_;
    } else if (_points.size() == 3) {
       T weights_ = FLT_MIN;
       for (int i = 0; i < _points[0].size(); i++) {
          T convexC_ij = this->convexityCriterion<T>(
             _points[0][i], _points[1][i], _normal[0][i], _normal[1][i]);
          T convexC_ic = this->convexityCriterion<T>(
             _points[0][i], _points[2][i], _normal[0][i], _normal[2][i]);
          T convexC_jc = this->convexityCriterion<T>(
             _points[1][i], _points[2][i], _normal[1][i], _normal[2][i]);

          // float angle_ = getVectorAngle(_normal[0][i], _normal[1][i]);
          // if (angle_ > ANGLE_THRESHOLD && convexC_ij <= 0) {
          //    convexC_ij =/ -1;
          // }
          weights_ = std::max(convexC_ij,
                               std::max(convexC_ic, convexC_jc));
       }
       return weights_;
    }
}

//----------------------------------------------------------------------
/**
 * return the weight of the point cloud
 */
/*template<typename T>
T RegionAdjacencyGraph::getCloudClusterWeightFunction(
    Eigen::Vector3f &center_point,
    Eigen::Vector3f &n1_point,
    Eigen::Vector3f &center_normal,
    Eigen::Vector3f &neigbour_normal) {
    T convexC = this->convexityCriterion<float>(
       center_point, n1_point, center_normal, neigbour_normal);

    T weight = convexC;
    return weight;
    }*/
//-----------------------------------------------------------------------


/**
 * returns the angle between 2 vectors
 */
float RegionAdjacencyGraph::getVectorAngle(
    const Eigen::Vector3f &vector1,
    const Eigen::Vector3f &vector2,
    bool indegree) {
    float angle_ = acos(vector1.dot(vector2));
    if (indegree) {
       return angle_ * 180/M_PI;
    } else {
       return angle_;
    }
}


/**
 * compute the cluster similarity and only cloud with convexity index
 * > 0 are assigned
 */
template<typename T>
T RegionAdjacencyGraph::convexityCriterion(
    const Eigen::Vector3f &center_point,
    const Eigen::Vector3f &n1_point,
    const Eigen::Vector3f &center_normal,
    const Eigen::Vector3f &neigbour_normal) {
    Eigen::Vector3f difference_ = center_point - n1_point;
    difference_ /= difference_.norm();
    T convexityc = static_cast<T>(center_normal.dot(difference_) -
                                  neigbour_normal.dot(difference_));
    return convexityc;
}

/**
 * 
 */
void RegionAdjacencyGraph::sampleRandomPointsFromCloudCluster(
    pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<pcl::Normal>::Ptr normal,
    std::vector<Eigen::Vector3f> &point_vector,
    std::vector<Eigen::Vector3f> &normal_vector,
    int gen_sz) {
    for (int i = 0; i < gen_sz; i++) {
       int _idx = rand() % cloud->size();
       Eigen::Vector3f cv = cloud->points[_idx].getVector3fMap();
       Eigen::Vector3f nv = Eigen::Vector3f(
          normal->points[_idx].normal_x,
          normal->points[_idx].normal_y,
          normal->points[_idx].normal_z);
       point_vector.push_back(cv);
       normal_vector.push_back(nv);
    }
}

/**
 * perform graph spliting, labeling and merging of similar regions
 */
void RegionAdjacencyGraph::splitMergeRAG(const int _threshold) {
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
           graph[*i].v_label = ++label;
        }
        AdjacencyIterator ai, a_end;
        tie(ai, a_end) = adjacent_vertices(*i, this->graph);
        for (; ai != a_end; ++ai) {
           bool found = false;
           EdgeDescriptor e_descriptor;
           tie(e_descriptor, found) = edge(*i, *ai, this->graph);
           if (found) {
              EdgeValue edge_val = boost::get(
                 boost::edge_weight, this->graph, e_descriptor);
              float weights_ = edge_val;
              if (weights_ < _threshold) {
                 remove_edge(e_descriptor, this->graph);
              } else {
                 if (this->graph[*ai].v_label == -1) {
                    this->graph[*ai].v_label = this->graph[*i].v_label;
                 }
              }
           }
        }
     }
#ifdef DEBUG
    // this->printGraph(this->graph);
    std::cout << MAGENTA << "\nPRINT INFO. \n --Graph Size: "
              << num_vertices(graph) << RESET <<
       std::endl << "--Total Label: " << label << "\n\n";
#endif  // DEBUG
}


/**
 * function to return the common neigbour between 2 cloud clusters
 */
int RegionAdjacencyGraph::getCommonNeigbour(
    const std::vector<int> &c1_neigbour,
    const std::vector<int> &c2_neigbour) {
    int commonIndex = -1;
    for (int j = 0; j < c1_neigbour.size(); j++) {
       int c1_val = c1_neigbour[j];
       for (int i = 0; i < c2_neigbour.size(); i++) {
          int c2_val = c2_neigbour[i];
          if (c1_val == c2_val) {
             commonIndex = c1_val;
             break;
          }
       }
    }
    return commonIndex;
}


/**
 * return the cluster labels as a vector of int
 */
void RegionAdjacencyGraph::getCloudClusterLabels(
    std::vector<int> &labelMD) {
    labelMD.clear();
    VertexIterator i, end;
    for (tie(i, end) = vertices(this->graph); i != end; ++i) {
       labelMD.push_back(static_cast<int>(this->graph[*i].v_label));
    }
}

/**
 * Print the tree
 */
void RegionAdjacencyGraph::printGraph(
    const Graph &_graph) {
    VertexIterator i, end;
    for (tie(i, end) = vertices(_graph); i != end; ++i) {
       AdjacencyIterator ai, a_end;
       tie(ai, a_end) = adjacent_vertices(*i, _graph);
       std::cout << *i << "\t" << _graph[*i].v_label << std::endl;
    }
}

/**
 * merging patches using the graph info
 */
void RegionAdjacencyGraph::concatenateRegionUsingRAGInfo(
    std::vector<pcl::PointCloud<PointT>::Ptr> &cloud_clusters,
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> &normal_clusters,
    std::map<int, pcl::PointCloud<PointT>::Ptr> &c_clusters) {
    if (cloud_clusters.empty() || normal_clusters.empty()) {
       ROS_ERROR("ERROR: Cannot Generate RAG of empty data...");
       return;
    }
    VertexIterator i, end;
    int icount = 0;
    for (tie(i, end) = vertices(this->graph); i != end; ++i) {
        int vertex_label = this->graph[*i].v_label;
        std::map<int, pcl::PointCloud<PointT>::Ptr>::iterator
           iter = c_clusters.find(vertex_label);
        if (iter == c_clusters.end()) {
           c_clusters.insert(
              std::map<int, pcl::PointCloud<PointT>::Ptr>::value_type(
                 vertex_label, cloud_clusters[icount]));
        } else if (iter != c_clusters.end()) {
           pcl::PointCloud<PointT>::Ptr m_cloud = (*iter).second;
           pcl::PointCloud<PointT>::Ptr c_cloud = cloud_clusters[icount];
           pcl::PointCloud<PointT>::Ptr n_cloud(new pcl::PointCloud<PointT>);
           this->mergePointCloud(m_cloud, c_cloud, n_cloud);
           (*iter).second = n_cloud;
        }
        icount++;
    }
    std::cout << "Map Size: " << c_clusters.size() << std::endl;
}



/*template<class T, class U>
void RegionAdjacencyGraph::mergePointCloud(
    const T m_cloud,
    const T c_cloud,
    T n_cloud) {
    n_cloud = T(new U);
    n_cloud->header = m_cloud->header;
    for (int i = 0; i < m_cloud->size(); i++) {
       n_cloud->push_back(m_cloud->points[i]);
    }
    for (int i = 0; i < c_cloud->size(); i++) {
       n_cloud->push_back(c_cloud->points[i]);
    }
    }*/


/**
 * merging 2 point cloud clusters of same sensor
 */
void RegionAdjacencyGraph::mergePointCloud(
    const pcl::PointCloud<PointT>::Ptr m_cloud,
    const pcl::PointCloud<PointT>::Ptr c_cloud,
    pcl::PointCloud<PointT>::Ptr n_cloud) {
    n_cloud->header = m_cloud->header;
    for (int i = 0; i < m_cloud->size(); i++) {
       PointT pt_ = m_cloud->points[i];
       if (!isnan(pt_.x) && !isnan(pt_.y) && !isnan(pt_.z)) {
          n_cloud->push_back(pt_);
       }
    }
    for (int i = 0; i < c_cloud->size(); i++) {
       PointT pt_ = c_cloud->points[i];
       if (!isnan(pt_.x) && !isnan(pt_.y) && !isnan(pt_.z)) {
          n_cloud->push_back(pt_);
       }
    }
}

// ----------------------------------------------------------------

/**
 * Compute RPY Histogram of each clusters
 */
void RegionAdjacencyGraph::computeCloudClusterRPYHistogram(
    const pcl::PointCloud<PointT>::Ptr _cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr _normal,
    cv::Mat &_histogram) {
    pcl::VFHEstimation<PointT,
                       pcl::Normal,
                       pcl::VFHSignature308> vfh;
    vfh.setInputCloud(_cloud);
    vfh.setInputNormals(_normal);
    pcl::search::KdTree<PointT>::Ptr tree(
       new pcl::search::KdTree<PointT>);
    vfh.setSearchMethod(tree);
    pcl::PointCloud<pcl::VFHSignature308>::Ptr _vfhs(
       new pcl::PointCloud<pcl::VFHSignature308>());
    vfh.compute(*_vfhs);
    _histogram = cv::Mat(sizeof(char), 308, CV_32F);
    for (int i = 0; i < _histogram.cols; i++) {
       _histogram.at<float>(0, i) = _vfhs->points[0].histogram[i];
    }

    float curvature_ = 0.0f;
    for (int i = 0; i < _normal->size(); i++) {
       curvature_ += _normal->points[i].curvature;
    }
    curvature_ /= static_cast<float>(_normal->size());
    //  _histogram *= curvature_;
    cv::normalize(_histogram, _histogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
}


