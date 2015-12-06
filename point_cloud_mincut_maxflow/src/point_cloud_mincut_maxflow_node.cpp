
#include <point_cloud_mincut_maxflow/point_cloud_mincut_maxflow.h>

PointCloudMinCutMaxFlow::PointCloudMinCutMaxFlow():
    num_threads_(8), neigbour_size_(8), is_search_(true) {
    pnh_.getParam("num_threads", this->num_threads_);
    pnh_.getParam("neigbour_size", this->neigbour_size_);
    pnh_.getParam("is_search", this->is_search_);

    this->onInit();
    this->subscribe();
}

void PointCloudMinCutMaxFlow::onInit() {
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/point_cloud_mincut_maxflow/output/cloud", 1);
    this->pub_obj_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/point_cloud_mincut_maxflow/output/selected_probability", 1);
    this->pub_indices_ = this->pnh_.advertise<
       jsk_recognition_msgs::ClusterPointIndices>(
          "/point_cloud_mincut_maxflow/output/indices", 1);
}

void PointCloudMinCutMaxFlow::subscribe() {
       this->sub_mask_.subscribe(this->pnh_, "in_mask", 1);
       this->sub_cloud_.subscribe(this->pnh_, "in_cloud", 1);
       this->sync_ = boost::make_shared<message_filters::Synchronizer<
          SyncPolicy> >(100);
       sync_->connectInput(sub_cloud_, sub_mask_);
       sync_->registerCallback(boost::bind(&PointCloudMinCutMaxFlow::callback,
                                           this, _1, _2));
}

void PointCloudMinCutMaxFlow::unsubscribe() {
    this->sub_cloud_.unsubscribe();
    this->sub_mask_.unsubscribe();
}

void PointCloudMinCutMaxFlow::callback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const sensor_msgs::PointCloud2::ConstPtr &mask_msg) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    pcl::PointCloud<PointT>::Ptr mask_cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*mask_msg, *mask_cloud);

    if (cloud->size() != mask_cloud->size()) {
       ROS_ERROR("ERROR: INPUT CLOUD AND MASK ARE NOT EQUAL");
       return;
    }

    // TODO(HERE): filter NAN Points and remove from Mask
    std::vector<int> removed_index;
    pcl::removeNaNFromPointCloud<PointT>(
       *cloud, *mask_cloud, removed_index);
    *cloud = *mask_cloud;
    
    
    std::cout << "DOING \t" << cloud->size() << std::endl;
    std::cout << num_threads_ << "\t" << neigbour_size_ << "\t"
              << is_search_ << std::endl;
    
    
    GraphPtr graph = GraphPtr(new Graph);
    // this->graph_ = GraphPtr(new Graph);
    this->makeAdjacencyGraph(graph, cloud, mask_cloud);

    std::cout << "\033[34m Graph Build \033[0m" << std::endl;
    
    ResidualCapacityMap residual_capacity = boost::get(
       boost::edge_residual_capacity, *graph);

    std::cout << "\033[34m Capacity Done \033[0m" << std::endl;
    std::cout << "S-T Info: " << source_ << "\t"
              << sink_ << std::endl;
    std::cout << "Graph Info: " << boost::num_edges(*graph) << "\t"
              << boost::num_vertices(*graph)<< std::endl;
    
    
    double max_flow = boost::boykov_kolmogorov_max_flow(
       *graph, this->source_, this->sink_);
    std::cout << "DONE: " << max_flow  << std::endl;
    
}

void PointCloudMinCutMaxFlow::makeAdjacencyGraph(
    GraphPtr graph,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<PointT>::Ptr mask_cloud) {
    if (cloud->empty() || mask_cloud->empty()) {
       ROS_ERROR("ERROR: EMPTY POINT CLOUD");
       return;
    }
    const int num_size = cloud->size() + 2;
    VertexDescriptor vertex_descriptor(0);
    this->vertices_.clear();
    this->vertices_.resize(num_size, vertex_descriptor);
    std::set<int> out_edges_marker;
    this->edge_marker_.clear();
    this->edge_marker_.resize(num_size, out_edges_marker);

    // TODO(HERE):  Make parallel
    for (int i = 0; i < num_size; i++) {
       this->vertices_[i] = boost::add_vertex(*graph);
    }

    this->source_ = this->vertices_[cloud->size()];
    this->sink_ = this->vertices_[cloud->size() + sizeof(char)];

    icounter = 0;
    
    //! get neigbours and add edge
    std::vector<float> neigbour_weights;
    if (this->is_search_) {
       bool is_search_ok = this->nearestNeigbourSearch(
          graph, mask_cloud, this->neigbour_size_, neigbour_weights);
       if (!is_search_ok || neigbour_weights.empty()) {
          ROS_ERROR("ERROR: NEIGBHOUR SEARCH FAILED");
          return;
       }
    }

    // TODO(HERE):  Make parallel
    for (int i = 0; i < cloud->size(); i++) {
       float src_weight = 0.80f;
       // float sink_weight = neigbour_weights[i];
       float sink_weight = 0.50f;
       this->addEdgeToGraph(graph, static_cast<int>(source_), i, src_weight);
       // this->addEdgeToGraph(graph, i, static_cast<int>(source_), src_weight);
       this->addEdgeToGraph(graph, static_cast<int>(sink_), i, sink_weight);
       // this->addEdgeToGraph(graph, i, static_cast<int>(sink_), sink_weight);
    }

}

bool PointCloudMinCutMaxFlow::nearestNeigbourSearch(
    GraphPtr graph, const pcl::PointCloud<PointT>::Ptr cloud,
    const int knearest, std::vector<float> &neigbour_weights) {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: EMPTY POINT CLOUD FOR KDTREE");
       return false;
    }
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);
    neigbour_weights.clear();
    neigbour_weights.reserve(cloud->size());
    float *nweight_ptr = &neigbour_weights[0];
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(this->num_threads_) shared(nweight_ptr)
// #endif
    for (int i = 0; i < cloud->size(); i++) {
       std::vector<int> point_idx_search;
       std::vector<float> point_squared_distance;
       PointT centroid_pt = cloud->points[i];
       if (!isnan(centroid_pt.x) || !isnan(centroid_pt.y) ||
           !isnan(centroid_pt.z)) {
          int search_out = kdtree.nearestKSearch(i
             /*centroid_pt*/, knearest, point_idx_search, point_squared_distance);
          float n_weights = 0.0f;

          std::cout << i << "\t";
          
          for (int j = 1; j < point_idx_search.size(); j++) {

             std::cout << point_idx_search[j] << ", ";
             
             
             // weight difference
             float weight = 1.0f - (
                abs(cloud->points[i].r - cloud->points[j].r) / 255.0f);
             this->addEdgeToGraph(graph, i, point_idx_search[j], weight);
             this->addEdgeToGraph(graph, point_idx_search[j], i, weight);
             n_weights += weight;
          }
          n_weights /= static_cast<float>(point_idx_search.size());
          // nweight_ptr[i] = n_weights;
          neigbour_weights.push_back(n_weights);

          std::cout  << std::endl;;
       }
    }
    return true;
}

void PointCloudMinCutMaxFlow::addEdgeToGraph(
    GraphPtr graph, const int source, const int sink, const float weight) {
    std::set<int>::iterator iter = this->edge_marker_[source].find(sink);
    if (iter != this->edge_marker_[source].end()) {
       // ROS_WARN("ERROR: EDGE NOT ADDED, %d \t %d", source, sink);
       return;
    } else {
       // ROS_INFO("INFO: EDGE ADDED, %d \t %d", source, sink);
    }
    EdgeDescriptor edge_des;
    EdgeDescriptor reverse_edge;
    bool edge_was_added;
    bool reverse_edge_was_added = true;
    boost::tie(edge_des, edge_was_added) = boost::add_edge(
       this->vertices_[source], this->vertices_[sink], *graph);
    boost::tie(reverse_edge, reverse_edge_was_added) = boost::add_edge(
       this->vertices_[sink], this->vertices_[source], *graph);
    if (!edge_was_added || !reverse_edge_was_added) {
       ROS_ERROR("ERROR: EDGE ADD FAILED");
       return;
    }

    // TODO(HERE): add other holders if required #L465
    this->edge_marker_[source].insert(sink);
}

void PointCloudMinCutMaxFlow::assembleLabels(
    std::vector<pcl::PointIndices> &clusters, GraphPtr graph,
    const ResidualCapacityMap &residual_capacity,
    const pcl::PointCloud<PointT>::Ptr cloud, const float epsilon) {
    std::vector<int> labels;
    labels.resize(cloud->size(), 0);
    for (int i = 0; i < cloud->size(); i++) {
       labels[i] = 1;
    }
    clusters.clear();
    pcl::PointIndices indices;
    clusters.resize(2, indices);
    OutEdgeIterator edge_iter;
    OutEdgeIterator edge_end;
    for (boost::tie(edge_iter, edge_end) = boost::out_edges(
            this->source_, *graph); edge_iter != edge_end; edge_iter++) {
       if (labels[edge_iter->m_target] == 1) {
          if (residual_capacity[*edge_iter] > epsilon) {
             clusters[1].indices.push_back(static_cast<int>(
                                              edge_iter->m_target));
          } else {
             clusters[0].indices.push_back(static_cast<int>(
                                              edge_iter->m_target));
          }
       }
    }
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "point_cloud_mincut_maxflow");
    PointCloudMinCutMaxFlow pcmm;
    ros::spin();
    return 0;
}
