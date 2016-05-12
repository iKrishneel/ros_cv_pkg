
#include <cuboid_bilateral_symmetric_segmentation/cuboid_bilateral_symmetric_segmentation.h>

CuboidBilateralSymmetricSegmentation::CuboidBilateralSymmetricSegmentation() :
    min_cluster_size_(50), leaf_size_(0.001f), symmetric_angle_thresh_(20) {
    this->occlusion_handler_ = boost::shared_ptr<OcclusionHandler>(
       new OcclusionHandler);
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    
    this->pnh_.getParam("leaf_size", this->leaf_size_);
    this->onInit();
}

void CuboidBilateralSymmetricSegmentation::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/cbss/output/cloud", 1);
    this->pub_edge_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/cbss/output/edges", 1);
    this->pub_indices_ = this->pnh_.advertise<jsk_msgs::ClusterPointIndices>(
        "/cbss/output/indices", 1);
    this->pub_bbox_ = this->pnh_.advertise<jsk_msgs::BoundingBoxArray>(
       "/cbss/output/bounding_box", 1);

    this->pub_normal_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/cbss/output/normals", 1);
}

void CuboidBilateralSymmetricSegmentation::subscribe() {
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_normal_.subscribe(this->pnh_, "input_normals", 1);
    this->sub_planes_.subscribe(this->pnh_, "input_planes", 1);
    this->sub_coef_.subscribe(this->pnh_, "input_coefficients", 1);
    
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
                                        SyncPolicy> >(100);
    this->sync_->connectInput(this->sub_cloud_, this->sub_normal_,
                              this->sub_planes_, this->sub_coef_);
    this->sync_->registerCallback(
        boost::bind(&CuboidBilateralSymmetricSegmentation::cloudCB,
                    this, _1, _2, _3, _4));
}

void CuboidBilateralSymmetricSegmentation::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_normal_.unsubscribe();
    this->sub_coef_.unsubscribe();
    this->sub_planes_.unsubscribe();
}

void CuboidBilateralSymmetricSegmentation::cloudCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::PointCloud2::ConstPtr &normal_msg,
    const jsk_msgs::PolygonArrayConstPtr &planes_msg,
    const jsk_msgs::ModelCoefficientsArrayConstPtr &coefficients_msg) {
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    pcl::PointCloud<NormalT>::Ptr normals (new pcl::PointCloud<NormalT>);
    pcl::fromROSMsg(*normal_msg, *normals);

    this->header_ = cloud_msg->header;
    
    SupervoxelMap supervoxel_clusters;
    pcl::PointCloud<NormalT>::Ptr sv_normals(new pcl::PointCloud<NormalT>);
    this->supervoxelDecomposition(supervoxel_clusters, sv_normals, cloud);

    // select the highest cluster
    float y_position = FLT_MAX;
    int start_index = -1;
    for (SupervoxelMap::const_iterator it = supervoxel_clusters.begin();
         it != supervoxel_clusters.end(); it++) {
       float y_pos = supervoxel_clusters.at(it->first)->centroid_.y;
       int csize = supervoxel_clusters.at(it->first)->voxels_->size();
       if (y_pos < y_position && csize > this->min_cluster_size_) {
          y_position = y_pos;
          start_index = it->first;
       }
    }
    if (start_index == -1) {
       ROS_ERROR("START ERROR");
       return;
    }

    this->kdtree_->setInputCloud(cloud);
    this->occlusion_handler_->setInputCloud(cloud);
    this->occlusion_handler_->setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    this->occlusion_handler_->initializeVoxelGrid();
    
    this->symmetryBasedObjectHypothesis(supervoxel_clusters, start_index,
                                        cloud, planes_msg, coefficients_msg);

    // publish supervoxel
    sensor_msgs::PointCloud2 ros_voxels;
    jsk_msgs::ClusterPointIndices ros_indices;
    this->publishSupervoxel(supervoxel_clusters,
                            ros_voxels, ros_indices, cloud_msg->header);
    this->pub_cloud_.publish(ros_voxels);
    this->pub_indices_.publish(ros_indices);
}

void CuboidBilateralSymmetricSegmentation::supervoxelDecomposition(
    SupervoxelMap &supervoxel_clusters, pcl::PointCloud<NormalT>::Ptr normals,
    const pcl::PointCloud<PointT>::Ptr cloud) {
    if (cloud->empty()) {
       ROS_ERROR("EMPTY CLOUD FOR SUPERVOXEL");
       return;
    }
    SupervoxelMap coplanar_supervoxels;
    AdjacencyList adjacency_list;
    this->supervoxelSegmentation(cloud,
                                 supervoxel_clusters,
                                 adjacency_list);
    UInt32Map voxel_labels;
    coplanar_supervoxels.clear();
    for (SupervoxelMap::iterator it = supervoxel_clusters.begin();
         it != supervoxel_clusters.end(); it++) {
       voxel_labels[it->first] = -1;
    }
        
    int label = -1;
    AdjacencyList::vertex_iterator i, end;
    for (boost::tie(i, end) = boost::vertices(adjacency_list); i != end; i++) {
       AdjacencyList::adjacency_iterator ai, a_end;
       boost::tie(ai, a_end) = boost::adjacent_vertices(*i, adjacency_list);
       uint32_t vindex = static_cast<int>(adjacency_list[*i]);
       UInt32Map::iterator it = voxel_labels.find(vindex);
       if (it->second == -1) {
          voxel_labels[vindex] = ++label;
       }
       bool vertex_has_neigbor = true;
       if (ai == a_end) {
          vertex_has_neigbor = false;
          if (!supervoxel_clusters.at(vindex)->voxels_->empty()) {
             coplanar_supervoxels[vindex] = supervoxel_clusters.at(vindex);
          }
       }

       Eigen::Vector4f v_normal = supervoxel_clusters.at(
          vindex)->normal_.getNormalVector4fMap();
       Eigen::Vector4f v_centroid = supervoxel_clusters.at(
          vindex)->centroid_.getVector4fMap();
       
       while (vertex_has_neigbor) {
          bool found = false;
          AdjacencyList::edge_descriptor e_descriptor;
          boost::tie(e_descriptor, found) = boost::edge(
             *i, *ai, adjacency_list);
          if (found) {
             float weight = adjacency_list[e_descriptor];
             uint32_t n_vindex = adjacency_list[*ai];

             Eigen::Vector4f n_normal = supervoxel_clusters.at(
               n_vindex)->normal_.getNormalVector4fMap();
             Eigen::Vector4f n_centroid = supervoxel_clusters.at(
                n_vindex)->centroid_.getVector4fMap();
             float coplanar_criteria = this->coplanarityCriteria(
                v_centroid, n_centroid, v_normal, n_normal,
                this->angle_threshold_, this->distance_threshold_);
             
             if (coplanar_criteria <= static_cast<float>(coplanar_threshold_)) {
                boost::remove_edge(e_descriptor, adjacency_list);
             } else {
                this->updateSupervoxelClusters(supervoxel_clusters,
                                               vindex, n_vindex);
                
                AdjacencyList::adjacency_iterator ni, n_end;
                boost::tie(ni, n_end) = boost::adjacent_vertices(
                   *ai, adjacency_list);
                
                for (; ni != n_end; ni++) {
                   bool is_found = false;
                   AdjacencyList::edge_descriptor n_edge;
                   boost::tie(n_edge, is_found) = boost::edge(
                      *ai, *ni, adjacency_list);
                   if (is_found && (*ni != *i)) {
                      boost::add_edge(*i, *ni, FLT_MIN, adjacency_list);
                   } else if (is_found && (*ni == *i)) {
                      continue;
                   }
                }
                boost::clear_vertex(*ai, adjacency_list);
                voxel_labels[n_vindex] = label;
             }
          }
          
          boost::tie(ai, a_end) = boost::adjacent_vertices(*i, adjacency_list);
          if (ai == a_end) {
             coplanar_supervoxels[vindex] = supervoxel_clusters.at(vindex);
             vertex_has_neigbor = false;
          } else {
             vertex_has_neigbor = true;
          }
       }
    }
    supervoxel_clusters.clear();
    supervoxel_clusters = coplanar_supervoxels;
}

bool CuboidBilateralSymmetricSegmentation::mergeNeigboringSupervoxels(
    SupervoxelMap &supervoxel_clusters, const int index) {
    if (supervoxel_clusters.empty() || index != -1) {
       ROS_ERROR("CANNOT MERGE DUE TO SIZE ERROR");
       return false;
    }
    //! find neigboring voxel in proximity
    double proximity = DBL_MAX;
    int idx = -1;
    Eigen::Vector4f centroid = supervoxel_clusters.at(
       index)->centroid_.getVector4fMap();
    for (SupervoxelMap::const_iterator it = supervoxel_clusters.begin();
         it != supervoxel_clusters.end(); it++) {
       double dist = pcl::distances::l2(
          centroid, it->second->centroid_.getVector4fMap());
       int vsize = it->second->voxels_->size();
       if (dist < proximity && vsize> this->min_cluster_size_) {
          proximity = dist;
          idx = it->first;
       }
    }
    if (idx == -1) {
       return false;
    }
    this->updateSupervoxelClusters(supervoxel_clusters, index, idx);
    return true;
}

float CuboidBilateralSymmetricSegmentation::coplanarityCriteria(
    const Eigen::Vector4f centroid, const Eigen::Vector4f n_centroid,
    const Eigen::Vector4f normal, const Eigen::Vector4f n_normal,
    const float angle_thresh, const float dist_thresh) {
    float tetha = std::acos(normal.dot(n_normal) / (
                              n_normal.norm() * normal.norm()));
    float ang_thresh = angle_thresh * (M_PI/180.0f);
    float coplannarity = 0.0f;
    if (tetha < ang_thresh) {
       float direct1 = normal.dot(centroid - n_centroid);
       float direct2 = n_normal.dot(centroid - n_centroid);
       float dist = std::fabs(std::max(direct1, direct2));
       if (dist < dist_thresh) {
          coplannarity = 1.0f;
       }
    }
    return coplannarity;
}

void CuboidBilateralSymmetricSegmentation::updateSupervoxelClusters(
    SupervoxelMap &supervoxel_clusters, const uint32_t vindex,
    const uint32_t n_vindex) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    *cloud = *(supervoxel_clusters.at(vindex)->voxels_) +
       *(supervoxel_clusters.at(n_vindex)->voxels_);
    pcl::PointCloud<pcl::Normal>::Ptr normals(
       new pcl::PointCloud<pcl::Normal>);
    *normals = *(supervoxel_clusters.at(vindex)->normals_) +
       *(supervoxel_clusters.at(n_vindex)->normals_);
    Eigen::Vector4f centre;
    pcl::compute3DCentroid<PointT, float>(*cloud, centre);
    pcl::PointXYZRGBA centroid;
    centroid.x = centre(0);
    centroid.y = centre(1);
    centroid.z = centre(2);
    pcl::PointXYZRGBA vcent = supervoxel_clusters.at(vindex)->centroid_;
    pcl::PointXYZRGBA n_vcent = supervoxel_clusters.at(n_vindex)->centroid_;
    centroid.g = (vcent.g - n_vcent.g)/2 + n_vcent.g;
    centroid.b = (vcent.b - n_vcent.b)/2 + n_vcent.b;
    centroid.a = (vcent.a - n_vcent.a)/2 + n_vcent.a;
    supervoxel_clusters.at(vindex)->voxels_->clear();
    supervoxel_clusters.at(vindex)->normals_->clear();
    *(supervoxel_clusters.at(vindex)->voxels_) = *cloud;
    *(supervoxel_clusters.at(vindex)->normals_) = *normals;
    supervoxel_clusters.at(vindex)->centroid_ = centroid;

    supervoxel_clusters.at(n_vindex)->voxels_->clear();
    supervoxel_clusters.at(n_vindex)->normals_->clear();
}

void CuboidBilateralSymmetricSegmentation::supervoxel3DBoundingBox(
    jsk_msgs::BoundingBox &bounding_box, pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<NormalT>::Ptr normals,
    const SupervoxelMap &supervoxel_clusters,
    const jsk_msgs::PolygonArrayConstPtr &planes_msg,
    const jsk_msgs::ModelCoefficientsArrayConstPtr &coefficients_msg,
    const int s_index) {
    if (supervoxel_clusters.empty() || s_index == -1) {
       ROS_ERROR("EMPTY VOXEL MAP");
       return;
    }
    cloud->clear();
    normals->clear();
    
    pcl::copyPointCloud<PointT, PointT>(
       *(supervoxel_clusters.at(s_index)->voxels_), *cloud);
    pcl::copyPointCloud<NormalT, NormalT>(
       *(supervoxel_clusters.at(s_index)->normals_), *normals);
    this->fitOriented3DBoundingBox(bounding_box, cloud, planes_msg,
                                   coefficients_msg);

    //! get bounding box symmetrical planes ? for debug
    pcl::PointCloud<PointT>::Ptr tmp_cloud(new pcl::PointCloud<PointT>);
    std::vector<Eigen::Vector4f> plane_coefficients;
    this->transformBoxCornerPoints(plane_coefficients, tmp_cloud, bounding_box);
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*tmp_cloud, ros_cloud);
    ros_cloud.header = planes_msg->header;
    this->pub_edge_.publish(ros_cloud);
}

void CuboidBilateralSymmetricSegmentation::symmetryBasedObjectHypothesis(
    SupervoxelMap &supervoxel_clusters, const int start_index,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const jsk_msgs::PolygonArrayConstPtr &planes_msg,
    const jsk_msgs::ModelCoefficientsArrayConstPtr &coefficients_msg) {
    if (supervoxel_clusters.empty() || cloud->empty()) {
       ROS_ERROR("EMPTY SUPERVOXEL FOR SYMMETRICAL CONSISTENCY");
       return;
    }
    jsk_msgs::BoundingBox bounding_box;
    pcl::PointCloud<PointT>::Ptr in_cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<NormalT>::Ptr in_normals(new pcl::PointCloud<NormalT>);
    this->supervoxel3DBoundingBox(
       bounding_box, in_cloud, in_normals, supervoxel_clusters,
       planes_msg, coefficients_msg, start_index);
    
    Eigen::Vector4f plane_coefficient;
    float max_energy = 0.0f;
    
    this->symmetricalConsistency(plane_coefficient, max_energy, in_cloud,
                                 in_normals, cloud, bounding_box);

    
    std::cout << plane_coefficient  << "\n\n";

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*in_cloud, ros_cloud);
    ros_cloud.header = planes_msg->header;
    this->pub_edge_.publish(ros_cloud);

    // publish bounding
    jsk_msgs::BoundingBoxArray bounding_boxes;
    bounding_box.header = planes_msg->header;
    bounding_boxes.boxes.push_back(bounding_box);
    bounding_boxes.header = planes_msg->header;
    pub_bbox_.publish(bounding_boxes);
}

bool CuboidBilateralSymmetricSegmentation::symmetricalConsistency(
    Eigen::Vector4f &plane_coefficient, float &max_energy,
    pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr normals,
    const pcl::PointCloud<PointT>::Ptr in_cloud,
    const jsk_msgs::BoundingBox bounding_box) {
    if (cloud->empty() || cloud->size() != normals->size() ||
        in_cloud->empty()) {
       ROS_ERROR("EMPTY CLOUD FOR SYMMETRICAL CONSISTENCY");
       return false;
    }

    pcl::PointCloud<PointT>::Ptr plane_points(new pcl::PointCloud<PointT>);
    std::vector<Eigen::Vector4f> plane_coefficients;
    this->transformBoxCornerPoints(plane_coefficients,
                                   plane_points, bounding_box);
    if (plane_coefficients.empty()) {
       ROS_ERROR("PLANE COEFFICENT NOT FOUND");
       return false;
    }
    double neigbor_dist_thresh = 0.02f;
    
    //! reflected point
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr symm_normal(
       new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    int best_plane = -1;
    // max_energy = 0.0f;
    pcl::PointCloud<PointT>::Ptr temp_cloud(new pcl::PointCloud<PointT>);
    for (int i = 0; i < plane_coefficients.size(); i++) {
       temp_cloud->clear();
       float symmetric_energy = 0.0f;
       Eigen::Vector3f plane_n = plane_coefficients[i].head<3>();
       for (int j = 0; j < cloud->size(); j++) {
          Eigen::Vector3f n = normals->points[j].getNormalVector3fMap();
          Eigen::Vector3f p = cloud->points[j].getVector3fMap();
          float beta = p(0)*plane_n(0) + p(1)*plane_n(1) + p(2)*plane_n(2);
          float alpha = (plane_n(0) * plane_n(0)) +
             (plane_n(1) * plane_n(1)) + (plane_n(2) * plane_n(2));
          float t = (plane_coefficients[i](3) - beta) / alpha;
          
          PointT pt = cloud->points[j];
          pt.x = p(0) + (t * 2 * plane_n(0));
          pt.y = p(1) + (t * 2 * plane_n(1));
          pt.z = p(2) + (t * 2 * plane_n(2));
          
          std::vector<int> neigbor_indices;
          this->getPointNeigbour<int>(neigbor_indices, pt, 1);
          int nidx = neigbor_indices[0];
          Eigen::Vector4f ref_pt = pt.getVector4fMap();
          ref_pt(3) = 0.0f;
          double distance = pcl::distances::l2(
             ref_pt, in_cloud->points[nidx].getVector4fMap());

          float weight = 0.0f;
          if (distance > neigbor_dist_thresh) {
             if (this->occlusionRegionCheck(pt)) {
                // WEIGHT = distance to plane
                float a = pcl::pointToPlaneDistance<PointT>(
                   pt, plane_coefficients[i]);
                float b = pcl::pointToPlaneDistance<PointT>(
                   cloud->points[j], plane_coefficients[i]);
                weight = 1.0f - (std::fabs(a - b) / 0.05f);
                
                pt.r = 0;
                pt.b = 255;
                pt.g = 255;
             } else {
                // SET TO BACKGROUND
                pt.r = 255;
                pt.b = 0;
                pt.g = 0;
             }
          } else {  //! get reflected normal
             Eigen::Vector3f symm_n = n -
                (2.0f*((plane_n.normalized()).dot(n))*plane_n.normalized());
             Eigen::Vector3f sneig_r = normals->points[
                nidx].getNormalVector3fMap();
             float dot_prod = (symm_n.dot(sneig_r)) / (
                symm_n.norm() * sneig_r.norm());
             float angle = std::acos(dot_prod) * (180.0f/M_PI);

             weight = 1.0 - (angle / 360.0f);
             
             if (angle > this->symmetric_angle_thresh_) {
                weight += (-1.0f * (this->symmetric_angle_thresh_ / 360.0f));
                
                pt.r = 0;
                pt.b = 0;
                pt.g = 255;
             } else {
                pt.r = 0;
                pt.b = 255;
                pt.g = 0;
             }
#ifdef RVIZ
             symm_normal->push_back(
                this->convertVector4fToPointXyzRgbNormal(
                   pt.getVector3fMap(),
                   symm_n, Eigen::Vector3f(0, 255, 0)));

             symm_normal->push_back(
                this->convertVector4fToPointXyzRgbNormal(
                   in_cloud->points[nidx].getVector3fMap(),
                   sneig_r, Eigen::Vector3f(255, 0, 0)));
                
             symm_normal->push_back(
                this->convertVector4fToPointXyzRgbNormal(
                   p, n, Eigen::Vector3f(0, 0, 255)));
#endif
          }
          temp_cloud->push_back(pt);

          if (!isnan(weight)) {
             symmetric_energy += weight;
          }
       }
       symmetric_energy /= static_cast<float>(cloud->size());
       if (symmetric_energy > max_energy) {
          best_plane = i;
          max_energy = symmetric_energy;
       }
       std::cout << "\033[34m TOTAL ENERY:\033[0m" << symmetric_energy << "\n";
    }
    plane_coefficient = plane_coefficients[best_plane];
    
    //! plot plane
    pcl::PointCloud<PointT>::Ptr plane(new pcl::PointCloud<PointT>);
    this->plotPlane(plane, plane_points, best_plane*3, (best_plane + 1) * 3);

    // cloud->clear();
    // *cloud += *plane;
    // *cloud += *temp_cloud;
    
    // just for viz
    sensor_msgs::PointCloud2 ros_normal;
    pcl::toROSMsg(*symm_normal, ros_normal);
    ros_normal.header = this->header_;
    this->pub_normal_.publish(ros_normal);
}

bool CuboidBilateralSymmetricSegmentation::occlusionRegionCheck(
    const PointT voxel) {
    Eigen::Vector3i grid_coord = Eigen::Vector3i();
    grid_coord = this->occlusion_handler_->getGridCoordinates(
       voxel.x, voxel.y, voxel.z);
    int state = INT_MAX;
    occlusion_handler_->occlusionEstimation(state, grid_coord);
    return (state == 0) ? false : true;
}

template<class T>
void CuboidBilateralSymmetricSegmentation::getPointNeigbour(
    std::vector<int> &neigbor_indices, const PointT seed_point,
    const T K, bool is_knn) {
    if (isnan(seed_point.x) || isnan(seed_point.y) || isnan(seed_point.z)) {
       ROS_ERROR("POINT IS NAN. RETURING VOID IN GET NEIGBOUR");
       return;
    }
    neigbor_indices.clear();
    std::vector<float> point_squared_distance;
    if (is_knn) {
       int search_out = this->kdtree_->nearestKSearch(
          seed_point, K, neigbor_indices, point_squared_distance);
    } else {
       int search_out = this->kdtree_->radiusSearch(
          seed_point, K, neigbor_indices, point_squared_distance);
    }
}

pcl::PointXYZRGBNormal
CuboidBilateralSymmetricSegmentation::convertVector4fToPointXyzRgbNormal(
    const Eigen::Vector3f &centroid, const Eigen::Vector3f &normal,
    const Eigen::Vector3f color) {
    pcl::PointXYZRGBNormal pt;
    pt.x = centroid(0);
    pt.y = centroid(1);
    pt.z = centroid(2);
    pt.r = color(0);
    pt.g = color(1);
    pt.b = color(2);
    pt.normal_x = normal(0);
    pt.normal_y = normal(1);
    pt.normal_z = normal(2);
    return pt;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "cuboid_bilateral_symmetric_segmentation");
    CuboidBilateralSymmetricSegmentation cbss;
    ros::spin();
    return 0;
}
