
#include <cuboid_bilateral_symmetric_segmentation/cuboid_bilateral_symmetric_segmentation.h>

CuboidBilateralSymmetricSegmentation::CuboidBilateralSymmetricSegmentation() {
    this->onInit();
}

void CuboidBilateralSymmetricSegmentation::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
      "cloud", 1);
    this->pub_edge_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "cloud2", 1);
     this->pub_indices_ = this->pnh_.advertise<
        jsk_msgs::ClusterPointIndices>("indices", 1);
}

void CuboidBilateralSymmetricSegmentation::subscribe() {
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_normal_.subscribe(this->pnh_, "input_normals", 1);
    this->sub_indices_.subscribe(this->pnh_, "input_indices", 1);
    this->sub_boxes_.subscribe(this->pnh_, "input_boxes", 1);
    
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
                                       SyncPolicy> >(100);
    this->sync_->connectInput(this->sub_cloud_,
                              // this->sub_indices_,
                              this->sub_normal_
                              // this->sub_boxes_
       );
    this->sync_->registerCallback(
        boost::bind(&CuboidBilateralSymmetricSegmentation::cloudCB,
                    this, _1, _2));
}

void CuboidBilateralSymmetricSegmentation::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_normal_.unsubscribe();
    this->sub_boxes_.unsubscribe();
    this->sub_indices_.unsubscribe();
}

void CuboidBilateralSymmetricSegmentation::cloudCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg2
    // const jsk_msgs::ClusterPointIndices::ConstPtr &indices_msg,
    // const jsk_msgs::BoundingBoxArray::ConstPtr &box_msg
   ) {
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    // std::vector<pcl::PointIndices::Ptr> indices =
    //    pcl_conversions::convertToPCLPointIndices(indices_msg->cluster_indices);
    // if (indices.size() != box_msg->boxes.size()) {
    //    ROS_ERROR("INDICES AND BOUNDING BOX ARRAY SIZE NOT EQUAL");
    //    return;
    // }

    SupervoxelMap supervoxel_clusters;
    pcl::PointCloud<NormalT>::Ptr normals(new pcl::PointCloud<NormalT>);
    this->supervoxelDecomposition(supervoxel_clusters, normals, cloud);

    std::cout << "CLUSTER: " << supervoxel_clusters.size()  << "\n";
    
    sensor_msgs::PointCloud2 ros_voxels;
    jsk_recognition_msgs::ClusterPointIndices ros_indices;
    this->publishSupervoxel(supervoxel_clusters,
                            ros_voxels, ros_indices, cloud_msg->header);
    this->pub_cloud_.publish(ros_voxels);
    this->pub_indices_.publish(ros_indices);
    
    // sensor_msgs::PointCloud2 ros_cloud;
    // pcl::toROSMsg(*cloud, ros_cloud);
    // ros_cloud.header = cloud_msg->header;
    // this->pub_cloud_.publish(ros_cloud);
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
    std::vector<uint32_t> good_vertices;
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
    // std::cout << tetha * (180/M_PI)  << "\t";
    if (tetha < ang_thresh) {
       float direct1 = normal.dot(centroid - n_centroid);
       float direct2 = n_normal.dot(centroid - n_centroid);
       float dist = std::fabs(std::max(direct1, direct2));
       // std::cout << dist << "\t" << dist_thresh << "\t" << ang_thresh;
       if (dist < dist_thresh) {
          coplannarity = 1.0f;
       }
    }
    // std::cout  << "\n";
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


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "cuboid_bilateral_symmetric_segmentation");
    CuboidBilateralSymmetricSegmentation cbss;
    ros::spin();
    return 0;
}

