
#ifndef _POINT_CLOUD_MINCUT_MAXFLOW_
#define _POINT_CLOUD_MINCUT_MAXFLOW_

#include <ros/ros.h>
#include <ros/console.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/features/normal_3d_omp.h>

#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
// #include <point_cloud_mincut_maxflow/graph.h>

#include <omp.h>
#include <string>

class PointCloudMinCutMaxFlow {
   
 private:
    boost::mutex mutex_;
    ros::NodeHandle pnh_;
    typedef message_filters::sync_policies::ApproximateTime<
       sensor_msgs::PointCloud2,
       sensor_msgs::PointCloud2> SyncPolicy;
   
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_mask_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    typedef boost::adjacency_list_traits<
       boost::vecS, boost::vecS, boost::directedS > Traits;
    typedef boost::property<boost::vertex_predecessor_t,
                            Traits::edge_descriptor> VertexPredecessor;
    typedef boost::property<boost::vertex_distance_t, long,
                           VertexPredecessor> VertexDistance;
    typedef boost::property<boost::vertex_color_t,
                           boost::default_color_type,
                           VertexDistance> VertexColor;
    typedef boost::property< boost::vertex_index_t, long,
                            VertexColor>  VertexIndex;
    typedef boost::property< boost::vertex_name_t, std::string,
                        VertexIndex > VertexName;
    typedef boost::property<boost::edge_reverse_t,
                           Traits::edge_descriptor > EdgeReverse;
    typedef boost::property<boost::edge_residual_capacity_t, float,
                           EdgeReverse> EdgeResidualCapacity;
    typedef boost::property<boost::edge_capacity_t, float,
                            EdgeResidualCapacity> EdgeCapacity;
    typedef boost::adjacency_list<boost::vecS, boost::vecS,
                                  boost::directedS, VertexName,
                                  EdgeCapacity> Graph;

    typedef boost::property_map<Graph,
                                boost::edge_capacity_t >::type CapacityMap;
    typedef boost::property_map<Graph,
                                boost::edge_reverse_t>::type ReverseEdgeMap;
    typedef Traits::vertex_descriptor VertexDescriptor;
    typedef boost::graph_traits<Graph>::edge_descriptor EdgeDescriptor;
    typedef boost::graph_traits<Graph>::out_edge_iterator OutEdgeIterator;
    typedef boost::graph_traits<Graph>::vertex_iterator VertexIterator;
    typedef boost::property_map<
       Graph, boost::edge_residual_capacity_t>::type ResidualCapacityMap;
    typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
    typedef boost::graph_traits< Graph>::in_edge_iterator InEdgeIterator;
    typedef boost::shared_ptr<Graph> GraphPtr;
    GraphPtr graph_;
   
    typedef typename boost::graph_traits<
       Graph>::adjacency_iterator AdjacencyIterator;
    boost::shared_ptr<CapacityMap> capacity_;
    boost::shared_ptr<ReverseEdgeMap> reverse_edges_;
   
    std::vector<VertexDescriptor> vertices_;
    std::vector<std::set<int> > edge_marker_;
    VertexDescriptor source_;
    VertexDescriptor sink_;
   
 protected:
    ros::Publisher pub_cloud_;
    ros::Publisher pub_obj_;
    ros::Publisher pub_indices_;

    void onInit();
    void subscribe();
    void unsubscribe();

    int num_threads_;
    int neigbour_size_;
    bool is_search_;

    int icounter;
   
 public:
    typedef pcl::PointXYZRGB PointT;
    PointCloudMinCutMaxFlow();
    void callback(
       const sensor_msgs::PointCloud2::ConstPtr &,
       const sensor_msgs::PointCloud2::ConstPtr &);
    bool nearestNeigbourSearch(
       const pcl::PointCloud<PointT>::Ptr, const int,
       std::vector<float> &);
    void makeAdjacencyGraph(
       const pcl::PointCloud<PointT>::Ptr,
       const pcl::PointCloud<PointT>::Ptr);
    void addEdgeToGraph(
       const int, const int, const float);
    void assembleLabels(
       std::vector<pcl::PointIndices> &, const GraphPtr,
       const ResidualCapacityMap &,
       const pcl::PointCloud<PointT>::Ptr, const float);
   
};


#endif  // _POINT_CLOUD_MINCUT_MAXFLOW_
