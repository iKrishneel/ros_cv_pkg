// Copyright (C) 2015 by Krishneel Chaudhary @ JSK Lab, The University of Tokyo

#include <point_cloud_scene_decomposer/region_adjacency_graph.h>
#include <map>
#include <utility>

/**
 * constructor 
 */
RegionAdjacencyGraph::RegionAdjacencyGraph() :
    comparision_points_size(100) {
   
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
    // this->comparision_points_size = 100;
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
           cv::Mat c_histogram;
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
             this->computeColorHistogram(cloud_clusters[j], c_histogram);
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
                cv::Mat nc_histogram;
                this->computeCloudClusterRPYHistogram(
                   cloud_clusters[n_index],
                   normal_clusters[n_index],
                   n_histogram);
                this->computeColorHistogram(
                   cloud_clusters[n_index], nc_histogram);

                distance = static_cast<float>(
                   cv::compareHist(
                      r_histogram, n_histogram, CV_COMP_BHATTACHARYYA));

                float c_dist = static_cast<float>(
                   cv::compareHist(
                      c_histogram, nc_histogram, CV_COMP_BHATTACHARYYA));
                
                // std::cout << "Distance " << distance  << "\t";
                
                distance = exp(-0.7 * distance) * exp(-0.5 * c_dist);
                
                // std::cout << "Prob: "  << distance << std::endl;
                
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
                 /*
                 std::cout << GREEN << "updating cloud ... " << GREEN
                           << RESET << std::endl;

                 // merge and update the cloud point
                 pcl::PointCloud<PointT>::Ptr m_cloud(
                     new pcl::PointCloud<PointT>);
                 pcl::PointCloud<pcl::Normal>::Ptr m_normal(
                     new pcl::PointCloud<pcl::Normal>);
                 this->mergePointCloud(
                     cloud_clusters[*i], cloud_clusters[neigbours_index],
                     normal_clusters[*i], normal_clusters[neigbours_index],
                     m_cloud, m_normal);

                 std::cout << "CLOUD SIZE: " << m_cloud->size() << std::endl;
                 
                 // insert to both merged location
                 cloud_clusters.at(*i) = m_cloud;
                 normal_clusters.at(*i) = m_normal;
                 cloud_clusters.at(neigbours_index) = m_cloud;
                 normal_clusters.at(neigbours_index) = m_normal;
                 
                 std::vector<Eigen::Vector3f> center_point;
                 std::vector<Eigen::Vector3f> center_normal;
                 cv::Mat normal_hist;
                 cv::Mat color_hist;
                 if (edge_weight_criteria == RAG_EDGE_WEIGHT_CONVEX_CRITERIA) {
                     this->sampleRandomPointsFromCloudCluster(
                         cloud_clusters[*i],
                         normal_clusters[*i],
                         center_point, center_normal,
                         this->comparision_points_size);
                     
                 } else if (edge_weight_criteria == RAG_EDGE_WEIGHT_DISTANCE) {
                     this->computeCloudClusterRPYHistogram(
                         m_cloud, m_normal, normal_hist);
                     this->computeColorHistogram(m_cloud, color_hist);
                 }

                 std::cout << GREEN << "updating graph... " << *ai << "\t"
                           << neigbours_index << "\t" << *i
                           << GREEN << RESET << std::endl;
                 
                 // get neigbours of ai
                 AdjacencyIterator ni, n_end;
                 // tie(ni, n_end) = boost::adjacent_vertices(*ai, this->graph);
                 tie(ni, n_end) = boost::adjacent_vertices(
                    neigbours_index, this->graph);
                 for (; ni != n_end; ni++) {

                    std::cout << YELLOW <<  "\t\t - N Neigobour: " << *ni
                              << YELLOW << RESET << std::endl;
                    
                    bool is_found = false;
                    EdgeDescriptor n_edge;
                    tie(n_edge, is_found) = boost::edge(
                       neigbours_index, *ni, this->graph);
                    if (is_found && (*ni != *i)) {
                       boost::add_edge(
                          *i, *ni, EdgeProperty(0.0f), this->graph);
                    }
                    boost::remove_edge(n_edge, this->graph);
                 }
                 boost::clear_vertex(neigbours_index, this->graph);

                 
                 // update the i's neighours
                 AdjacencyIterator ii, i_end;
                 tie(ii, i_end) = adjacent_vertices(*i, this->graph);
                 for (; ii != i_end; ii++) {
                    bool is_found = false;
                    EdgeDescriptor i_edge;
                    tie(i_edge, is_found) = edge(*i, *ii, this->graph);
                    if (is_found) {
                        float e_weight = 0.0f;
                        if (edge_weight_criteria ==
                            RAG_EDGE_WEIGHT_CONVEX_CRITERIA) {
                            std::vector<Eigen::Vector3f> n1_point;
                            std::vector<Eigen::Vector3f> n1_normal;
                            this->sampleRandomPointsFromCloudCluster(
                                cloud_clusters[*ii],
                                normal_clusters[*ii],
                                n1_point, n1_normal,
                                this->comparision_points_size);
                            
                            std::vector<std::vector<Eigen::Vector3f> > _points;
                            std::vector<std::vector<Eigen::Vector3f> > _normals;
                            _points.push_back(center_point);
                            _points.push_back(n1_point);
                            _normals.push_back(center_normal);
                            _normals.push_back(n1_normal);
                            e_weight =
                               this->getCloudClusterWeightFunction<float>(
                                  _points, _normals);
                        } else if (edge_weight_criteria ==
                                   RAG_EDGE_WEIGHT_DISTANCE) {
                            cv::Mat nhist;
                            cv::Mat chist;
                            this->computeCloudClusterRPYHistogram(
                                c_clusters[*ii], n_clusters[*ii], nhist);
                            this->computeColorHistogram(c_clusters[*ii], chist);
                            e_weight = this->getEdgeWeight(
                                normal_hist, nhist, color_hist, chist);
                        }
                       std::cout << "weight: " << e_weight << "\t"
                                 << *ii << std::endl;
                       
                       remove_edge(i_edge, this->graph);
                       boost::add_edge(
                             *i, *ii, EdgeProperty(e_weight), this->graph);
                    }
                 }
                 */
              }
           }
           /*
           //  check if neigbor exits
           tie(ai, a_end) = boost::adjacent_vertices(*i, this->graph);

           if (ai == a_end) {
              vertex_has_neigbor = false;
           }
           */         
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





float RegionAdjacencyGraph::getEdgeWeight(
    const cv::Mat &rpy_1, const cv::Mat &rpy_2,
    const cv::Mat &color_1, const cv::Mat &color_2) {
    float rpy_dist = static_cast<float>(
        cv::compareHist(rpy_1, rpy_2, CV_COMP_BHATTACHARYYA));
    float color_dist = static_cast<float>(
        cv::compareHist(color_1, color_2, CV_COMP_BHATTACHARYYA));
    float prob = exp(-0.7 * rpy_dist) * exp(-0.5 * color_dist);
    return prob;
}


/**
 * merging 2 point cloud clusters of same sensor
 */
void RegionAdjacencyGraph::mergePointCloud(
    const pcl::PointCloud<PointT>::Ptr cloud_1,
    const pcl::PointCloud<PointT>::Ptr cloud_2,
    const pcl::PointCloud<pcl::Normal>::Ptr normal_1,
    const pcl::PointCloud<pcl::Normal>::Ptr normal_2,
    pcl::PointCloud<PointT>::Ptr out_cloud,
    pcl::PointCloud<pcl::Normal>::Ptr out_normal) {
    out_cloud->header = cloud_1->header;
    out_normal->header = normal_1->header;
    for (int i = 0; i < cloud_1->size(); i++) {
        PointT pt_ = cloud_1->points[i];
        pcl::Normal nrm = normal_1->points[i];
        if (!isnan(pt_.x) && !isnan(pt_.y) && !isnan(pt_.z)) {
            out_cloud->push_back(pt_);
            out_normal->push_back(nrm);
        }
    }
    for (int i = 0; i < cloud_2->size(); i++) {
        PointT pt_ = cloud_2->points[i];
        pcl::Normal nrm = normal_2->points[i];
        if (!isnan(pt_.x) && !isnan(pt_.y) && !isnan(pt_.z)) {
            out_cloud->push_back(pt_);
            out_normal->push_back(nrm);
        }
    }
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
    // cv::normalize(_histogram, _histogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
}


void RegionAdjacencyGraph::computeColorHistogram(
    const pcl::PointCloud<PointT>::Ptr _cloud,
    cv::Mat &hist, const int hBin, const int sBin, bool is_norm) {
    cv::Mat pixels = cv::Mat::zeros(
       sizeof(char), static_cast<int>(_cloud->size()), CV_8UC3);
    for (int i = 0; i < _cloud->size(); i++) {
       cv::Vec3b pix_val;
       pix_val[0] = _cloud->points[i].b;
       pix_val[1] = _cloud->points[i].g;
       pix_val[2] = _cloud->points[i].r;
       pixels.at<cv::Vec3b>(0, i) = pix_val;
    }
    cv::Mat hsv;
    cv::cvtColor(pixels, hsv, CV_BGR2HSV);
    int histSize[] = {hBin, sBin};
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    const float* ranges[] = {h_ranges, s_ranges};
    int channels[] = {0, 1};
    cv::calcHist(
       &hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    if (is_norm) {
       cv::normalize(hist, hist, 0, 1, cv::NORM_L2, -1, cv::Mat());
    }
}
