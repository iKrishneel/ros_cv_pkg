
#include <convex_connected_voxels/convex_connected_voxels.h>

#include <map>
#include <vector>

ConvexConnectedVoxels::ConvexConnectedVoxels() {
    this->onInit();
}

void ConvexConnectedVoxels::onInit() {
    this->subscribe();

    this->pub_cloud_ =  this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/convex_connected_voxels/output/cloud", 1);

    this->pub_indices_ = this->pnh_.advertise<
       jsk_recognition_msgs::ClusterPointIndices>(
          "/convex_connected_voxels/output/indices", 1);
}

void ConvexConnectedVoxels::subscribe() {
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_normal_.subscribe(this->pnh_, "input_normal", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
                                        SyncPolicy> >(100);
    this->sync_->connectInput(sub_cloud_, sub_normal_);
    this->sync_->registerCallback(
       boost::bind(&ConvexConnectedVoxels::callback, this, _1, _2));
}

void ConvexConnectedVoxels::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_normal_.unsubscribe();
}

void ConvexConnectedVoxels::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::PointCloud2::ConstPtr &normal_msg) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
    this->surfelLevelObjectHypothesis(cloud, normals, supervoxel_clusters);
    sensor_msgs::PointCloud2 ros_voxels;
    jsk_recognition_msgs::ClusterPointIndices ros_indices;
    this->publishSupervoxel(supervoxel_clusters,
                            ros_voxels, ros_indices, cloud_msg->header);
    this->pub_cloud_.publish(ros_voxels);
    this->pub_indices_.publish(ros_indices);
}

void ConvexConnectedVoxels::surfelLevelObjectHypothesis(
    pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<pcl::Normal>::Ptr normals,
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > &convex_supervoxels) {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: EMPTY CLOUD");
       return;
    }
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
    AdjacencyList adjacency_list;
    this->supervoxelSegmentation(cloud,
                                 supervoxel_clusters,
                                 adjacency_list);
    std::map<uint32_t, int> voxel_labels;
    convex_supervoxels.clear();
    // cloud->clear();
    for (std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>::iterator it =
            supervoxel_clusters.begin(); it != supervoxel_clusters.end();
         it++) {
       voxel_labels[it->first] = -1;
       // pcl::Supervoxel<PointT>::Ptr supervoxel =
       //    supervoxel_clusters.at(it->first);
    }
    int label = -1;
    AdjacencyList::vertex_iterator i, end;

    /*
    std::cout << supervoxel_clusters.size()  << "\n\n";
    for (boost::tie(i, end) = boost::vertices(adjacency_list); i != end; i++) {
       AdjacencyList::adjacency_iterator ai, a_end;
       boost::tie(ai, a_end) = boost::adjacent_vertices(*i, adjacency_list);
       std::cout << adjacency_list[*i]  << "\t ->";
       for (; ai != a_end; ai++) {
          std::cout << adjacency_list[*ai]  << ", ";
       }
       std::cout  << "\n";
    }
    ROS_WARN("DONE\n\n");
    return;
    */
    
    std::vector<uint32_t> good_vertices;
    for (boost::tie(i, end) = boost::vertices(adjacency_list); i != end; i++) {
       AdjacencyList::adjacency_iterator ai, a_end;
       boost::tie(ai, a_end) = boost::adjacent_vertices(*i, adjacency_list);
       uint32_t vindex = static_cast<int>(adjacency_list[*i]);
       
       // Eigen::Vector4f v_normal = this->cloudMeanNormal(
       //    supervoxel_clusters.at(vindex)->normals_);
       Eigen::Vector4f v_normal = supervoxel_clusters.at(
          vindex)->normal_.getNormalVector4fMap();
       std::map<uint32_t, int>::iterator it = voxel_labels.find(vindex);
       if (it->second == -1) {
          voxel_labels[vindex] = ++label;
       }

       bool vertex_has_neigbor = true;
       if (ai == a_end) {
          vertex_has_neigbor = false;

          std::cout << "SKIP CHECKING : "<< vindex << "\t"
                    << vertex_has_neigbor  << "\n";
          
         if (!supervoxel_clusters.at(vindex)->voxels_->empty()) {
            convex_supervoxels[vindex] = supervoxel_clusters.at(vindex);
         }
       }
       
       std::vector<uint32_t> neigb_ind;
       while (vertex_has_neigbor) {
          bool found = false;
          AdjacencyList::edge_descriptor e_descriptor;
          boost::tie(e_descriptor, found) = boost::edge(
             *i, *ai, adjacency_list);
          if (found) {
             float weight = adjacency_list[e_descriptor];
             uint32_t n_vindex = adjacency_list[*ai];

             std::cout << "processing: " << n_vindex  << "\t"
                       << supervoxel_clusters.size() << "\n";

             Eigen::Vector4f update_normal = this->cloudMeanNormal(
                supervoxel_clusters.at(vindex)->normals_, true);
             Eigen::Vector4f update_centroid;
             pcl::compute3DCentroid<PointT, float>(
                *(supervoxel_clusters.at(vindex)->voxels_), update_centroid);

             
             Eigen::Vector4f dist = (
                supervoxel_clusters.at(vindex)->centroid_.getVector4fMap() -
                supervoxel_clusters.at(n_vindex)->centroid_.getVector4fMap())/
                (supervoxel_clusters.at(vindex)->centroid_.getVector4fMap() -
                 supervoxel_clusters.at(n_vindex)->centroid_.getVector4fMap()
                   ).norm();
             
             /*
             Eigen::Vector4f dist = (
                update_centroid - supervoxel_clusters.at(
                   n_vindex)->centroid_.getVector4fMap()) / (
                      update_centroid - supervoxel_clusters.at(
                         n_vindex)->centroid_.getVector4fMap()).norm();
             */
             
             float conv_criteria = (
                // this->cloudMeanNormal(supervoxel_clusters.at(vindex)->normals_) -
                // supervoxel_clusters.at(vindex)->normal_.getNormalVector4fMap()-
                update_normal  -
                supervoxel_clusters.at(n_vindex)->normal_.getNormalVector4fMap()
                ).dot(dist);

             // conv_criteria = (supervoxel_clusters.at(
             //                     n_vindex)->centroid_.getVector4fMap() -
             //    update_centroid).dot(supervoxel_clusters.at(
             // n_vindex)->normal_.getNormalVector4fMap());
                
             
             neigb_ind.push_back(n_vindex);
             if (conv_criteria <= static_cast<float>(this->convex_threshold_) ||
                 isnan(conv_criteria)) {
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
                      // boost::remove_edge(n_edge, adjacency_list);
                   } else if (is_found && (*ni == *i)) {
                      // boost::remove_edge(n_edge, adjacency_list);
                   }

                   std::cout << "\t\tREMVING: " << adjacency_list[*ni]  << "\t"
                             << n_vindex << "\t" << vindex << "\t"
                             << adjacency_list[*ai] << "\n";
                   
                }
                ROS_WARN("REMOVING");

                boost::clear_vertex(*ai, adjacency_list);
                // std::cout << adjacency_list[boost::source(
                //       e_descriptor, adjacency_list)]  << "\n";
                // std::cout << adjacency_list[boost::target(
                //       e_descriptor, adjacency_list)]  << "\n";


                // boost::clear_vertex(v1, adjacency_list);
                
                ROS_WARN("ADDING");
                voxel_labels[n_vindex] = label;
             }
          }

          ROS_INFO("UPDATING");
          
          boost::tie(ai, a_end) = boost::adjacent_vertices(*i, adjacency_list);
          if (ai == a_end) {
             convex_supervoxels[vindex] = supervoxel_clusters.at(vindex);
            vertex_has_neigbor = false;
          } else {
             vertex_has_neigbor = true;
          }
       }
       if (ai == a_end) {
          // convex_supervoxels[vindex] = supervoxel_clusters.at(vindex);
       }
    }
    // convex_supervoxels.clear();
    // std::vector<bool> flag;
    // for (int i = 0; i < label; i++) {
    //    flag.push_back(false);
    // }
    // for (std::map<uint32_t, int>::iterator it = voxel_labels.begin();
    //      it != voxel_labels.end(); it++) {
    //    if (!flag[i])
    //    convex_supervoxels[]
    //    std::cout << it->first << ", " << it->second  << "\n";
    // }


    
    std::cout << "\033[31m LABEL: \033[0m"  << label << "\n";
    std::cout << convex_supervoxels.size()  << "\t" << good_vertices.size()
              << "\n\n";
    supervoxel_clusters.clear();
}

void ConvexConnectedVoxels::updateSupervoxelClusters(
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> &supervoxel_clusters,
    const uint32_t vindex, const uint32_t n_vindex) {
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
    // pcl::PointXYZRGBA centroid;
    pcl::PointXYZRGBA vcent = supervoxel_clusters.at(vindex)->centroid_;
    pcl::PointXYZRGBA n_vcent = supervoxel_clusters.at(n_vindex)->centroid_;
    // centroid.x = (vcent.x - n_vcent.x)/2 + n_vcent.x;
    // centroid.y = (vcent.y - n_vcent.y)/2 + n_vcent.y;
    // centroid.z = (vcent.z - n_vcent.z)/2 + n_vcent.z;
    // centroid.r = (vcent.r - n_vcent.r)/2 + n_vcent.r;
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
    // *(supervoxel_clusters.at(n_vindex)->voxels_) = *cloud;
    // *(supervoxel_clusters.at(n_vindex)->normals_) = *normals;
    // supervoxel_clusters.at(n_vindex)->centroid_ = centroid;

    // std::cout << "SIZE: " << supervoxel_clusters.at(n_vindex)->voxels_->size() << ", "
    //           << n_vindex << "\n";
}

Eigen::Vector4f ConvexConnectedVoxels::cloudMeanNormal(
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
        1.0f);
    if (isnorm) {
        n_mean.normalize();
    }
    return n_mean;
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "convex_connected_voxels");
    ConvexConnectedVoxels ccv;
    ros::spin();
    return 0;
}
