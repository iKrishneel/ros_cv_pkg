
#include <point_cloud_mincut_maxflow/point_cloud_mincut_maxflow.h>

PointCloudMinCutMaxFlow::PointCloudMinCutMaxFlow() {
    pnh_.getParam("num_threads", this->num_threads_);
    pnh_.getParam("neigbour_size", this->neigbour_size_);
    pnh_.getParam("is_search", this->is_search_);
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

    GraphPtr graph = GraphPtr(new Graph);
    this->makeAdjacencyGraph(graph, cloud, mask_cloud);

    ResidualCapacityMap residual_capacity = boost::get(
       boost::edge_residual_capacity, *graph);

    double max_flow = boost::boykov_kolmogorov_max_flow(
       *graph, this->source_, this->sink_);

    
    
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

    //! get neigbours and add edge
    std::vector<pcl::PointIndices> neigbour_indices;
    if (this->is_search_) {
       if (!this->nearestNeigbourSearch(
              graph, mask_cloud, this->neigbour_size_, neigbour_indices)) {
          return;
       }
    }

    if (neigbour_indices.empty()) {
       return;
    }

    // TODO(HERE):  Make parallel
    for (int i = 0; i < cloud->size(); i++) {
       float src_weight = 0.0;
       float sink_weight = 0.0;

       // TODO(.): HERE
       // sink and src weight get neigbours and average the weight

       this->addEdgeToGraph(graph, static_cast<int>(source_), i, src_weight);
       this->addEdgeToGraph(graph, static_cast<int>(sink_), i, sink_weight);
    }
}

bool PointCloudMinCutMaxFlow::nearestNeigbourSearch(
    GraphPtr graph, const pcl::PointCloud<PointT>::Ptr cloud,
    const int knearest, std::vector<pcl::PointIndices> &neigbour_indices) {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: EMPTY POINT CLOUD FOR KDTREE");
       return false;
    }
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);
    neigbour_indices.reserve(cloud->size());
    pcl::PointIndices *indices_ptr = &neigbour_indices[0];
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_) shared(indices_ptr)
#endif
    for (int i = 0; i < cloud->size(); i++) {
       std::vector<int> point_idx_search;
       std::vector<float> point_squared_distance;
       PointT centroid_pt = cloud->points[i];
       if (!isnan(centroid_pt.x) || !isnan(centroid_pt.y) ||
           !isnan(centroid_pt.z)) {
          int search_out = kdtree.nearestKSearch(
             centroid_pt, knearest, point_idx_search, point_squared_distance);
          float n_weights = 0.0f;
          for (int j = 0; j < point_idx_search.size(); j++) {
             float weight = cloud->points[j].r;
             this->addEdgeToGraph(graph, i, point_idx_search[j], weight);
             this->addEdgeToGraph(graph, point_idx_search[j], i, weight);
             n_weights += weight;
          }
          n_weights /= static_cast<float>(point_idx_search.size());
       }
       indices_ptr[i].indices = point_idx_search;
    }
    return true;
}

void PointCloudMinCutMaxFlow::addEdgeToGraph(
    GraphPtr graph, const int source, const int sink, const float weight) {
    std::set<int>::iterator iter = this->edge_marker_[source].find(sink);
    if (iter != this->edge_marker_[source].end()) {
       return;
    }
    EdgeDescriptor edge_des;
    EdgeDescriptor reverse_edge;
    bool edge_was_added;
    bool reverse_edge_was_added;
    boost::tie(edge_des, edge_was_added) = boost::add_edge(
       this->vertices_[source], this->vertices_[sink], *graph);
    boost::tie(reverse_edge, reverse_edge_was_added) = boost::add_edge(
       this->vertices_[sink], this->vertices_[source], *graph);
    if (!edge_was_added || !reverse_edge_was_added) {
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
