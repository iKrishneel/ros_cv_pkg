
#include <cuboid_bilateral_symmetric_segmentation/cuboid_bilateral_symmetric_segmentation.h>

CuboidBilateralSymmetricSegmentation::CuboidBilateralSymmetricSegmentation() :
    min_cluster_size_(50), leaf_size_(0.01f) {
    this->occlusion_handler_ = boost::shared_ptr<OcclusionHandler>(
      new OcclusionHandler);
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
    
    SupervoxelMap supervoxel_clusters;
    pcl::PointCloud<NormalT>::Ptr sv_normals(new pcl::PointCloud<NormalT>);
    this->supervoxelDecomposition(supervoxel_clusters, sv_normals, cloud);

    std::cout << "CLUSTER: " << supervoxel_clusters.size()  << "\n";
    
    jsk_msgs::BoundingBox bounding_box;
    this->supervoxel3DBoundingBox(bounding_box, supervoxel_clusters,
                                  planes_msg, coefficients_msg, 1);

    // publish supervoxel
    sensor_msgs::PointCloud2 ros_voxels;
    jsk_msgs::ClusterPointIndices ros_indices;
    this->publishSupervoxel(supervoxel_clusters,
                            ros_voxels, ros_indices, cloud_msg->header);
    this->pub_cloud_.publish(ros_voxels);
    this->pub_indices_.publish(ros_indices);

    // publish bounding
    jsk_msgs::BoundingBoxArray bounding_boxes;
    bounding_box.header = cloud_msg->header;
    bounding_boxes.boxes.push_back(bounding_box);
    bounding_boxes.header = cloud_msg->header;
    pub_bbox_.publish(bounding_boxes);

    
    std::clock_t start;
    double duration;
    start = std::clock();
    
    this->symmetricalConsistency(cloud, normals, bounding_box);

    duration = (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
    ROS_WARN("DURATION: %f", duration);
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_edge_.publish(ros_cloud);
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
    jsk_msgs::BoundingBox &bounding_box,
    const SupervoxelMap &supervoxel_clusters,
    const jsk_msgs::PolygonArrayConstPtr &planes_msg,
    const jsk_msgs::ModelCoefficientsArrayConstPtr &coefficients_msg,
    const int index) {

    ROS_INFO("\033[34mCOMPUTING BOUNDING BOX \33[0m");
   
    if (supervoxel_clusters.empty() || index > supervoxel_clusters.size()) {
       ROS_ERROR("EMPTY VOXEL MAP");
       return;
    }
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    // select the highest cluster
    float y_position = FLT_MAX;
    int s_index = -1;
    for (SupervoxelMap::const_iterator it = supervoxel_clusters.begin();
         it != supervoxel_clusters.end(); it++) {
       float y_pos = supervoxel_clusters.at(it->first)->centroid_.y;
       int csize = supervoxel_clusters.at(it->first)->voxels_->size();
       ROS_WARN("SIZE: %d", csize);
       if (y_pos < y_position && csize > this->min_cluster_size_) {
          y_position = y_pos;
          s_index = it->first;
       }
       // *cloud += *(supervoxel_clusters.at(it->first)->voxels_);
    }
    if (s_index == -1) {
       ROS_ERROR("CLUSTERS ARE TOO SMALL");
       return;
    }
    pcl::copyPointCloud<PointT, PointT>(
       *(supervoxel_clusters.at(s_index)->voxels_), *cloud);
    this->fitOriented3DBoundingBox(bounding_box, cloud, planes_msg,
                                   coefficients_msg);

    //! get bounding box symmetrical planes
    cloud->clear();
    std::vector<Eigen::Vector4f> plane_coefficients;
    this->transformBoxCornerPoints(plane_coefficients, cloud, bounding_box);
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = planes_msg->header;
    this->pub_edge_.publish(ros_cloud);
}

bool CuboidBilateralSymmetricSegmentation::symmetricalConsistency(
    pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<NormalT>::Ptr normals,
    const jsk_msgs::BoundingBox bounding_box) {
    if (cloud->empty() || cloud->size() != normals->size()) {
       ROS_ERROR("EMPTY CLOUD FOR SYMMETRICAL CONSISTENCY");
       return false;
    }
    this->occlusion_handler_->setInputCloud(cloud);
    this->occlusion_handler_->setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    this->occlusion_handler_->initializeVoxelGrid();
    
    pcl::PointCloud<PointT>::Ptr temp_cloud(new pcl::PointCloud<PointT>);
    this->occlusion_handler_->filter(*temp_cloud);
    cloud->clear();
    *cloud = *temp_cloud;
    
    std::vector<Eigen::Vector4f> plane_coefficients;
    this->transformBoxCornerPoints(plane_coefficients,
                                   temp_cloud, bounding_box);
    if (plane_coefficients.empty()) {
       ROS_ERROR("PLANE COEFFICENT NOT FOUND");
       return false;
    }

    //! plot plane
    pcl::PointCloud<PointT>::Ptr plane(new pcl::PointCloud<PointT>);
    this->plotPlane(plane, temp_cloud, 3*3);
    
    //! reflected point
    temp_cloud->clear();
    for (int i = 0; i < 1; i++) {
       i = 3;
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
          
          if (this->occlusionRegionCheck(pt)) {
             pt.r = 0;
             pt.b = 255;
             pt.g = 255;
          } else {
             pt.r = 0;
             pt.b = 255;
             pt.g = 0;
          }
          temp_cloud->push_back(pt);
       }
    }
    cloud->clear();
    *cloud += *temp_cloud;
    *cloud += *plane;
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

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "cuboid_bilateral_symmetric_segmentation");
    CuboidBilateralSymmetricSegmentation cbss;
    ros::spin();
    return 0;
}
