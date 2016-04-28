
#include <dynamic_state_segmentation/graphcuts_optimization.h>

GraphCutsOptimization::GraphCutsOptimization(
    const int seed_index, const std::vector<int> seed_neigbors,
    const std::vector<std::vector<int> > neigbor_indices,
    const pcl::PointCloud<PointT>::Ptr weight_map,
    const pcl::PointCloud<PointT>::Ptr cloud) :
    seed_neigbors_(seed_neigbors),
    neigbor_indices_(neigbor_indices),
    seed_index_(seed_index) {    
}

void GraphCutsOptimization::dynamicSegmentation(
    const pcl::PointCloud<PointT>::Ptr weight_map,
    const pcl::PointCloud<PointT>::Ptr cloud) {
    if (cloud->empty()) {
       ROS_ERROR("INCORRECT SIZE FOR DYNAMIC STATE");
       return;
    }
    
    ROS_INFO("\033[34m GRAPH CUT \033[0m");
    
    const int node_num = cloud->size();
    const int edge_num = 8;
    boost::shared_ptr<GraphType> graph(new GraphType(
                                          node_num, edge_num * node_num));
    
    for (int i = 0; i < node_num; i++) {
       graph->add_node();
    }

    ROS_INFO("\033[34m GRAPH INIT %d \033[0m", node_num);
    
    // pcl::PointCloud<PointT>::Ptr weight_map(new pcl::PointCloud<PointT>);
    // std::vector<std::vector<int> > neigbor_indices;
    // this->potentialFunctionKernel(neigbor_indices, weight_map, cloud, normals);

    ROS_INFO("\033[34m POTENTIAL COMPUTED \033[0m");

    // get seed_region neigbours for hard label
    // const float s_radius = 0.02f;
    // this->getPointNeigbour<float>(seed_neigbors_, cloud, this->seed_point_, s_radius, false);

    std::vector<bool> label_cache(static_cast<int>(cloud->size()));
    for (int i = 0; i < cloud->size(); i++) {
	label_cache[i] = false;
    }

    for (int i = 0; i < this->seed_neigbors_.size(); i++) {
	int index = this->seed_neigbors_[i];
	graph->add_tweights(index, HARD_THRESH, 0);
	label_cache[index] = true;
    }
    
    for (int i = 0; i < weight_map->size(); i++) {
       float weight = weight_map->points[i].r;
       float edge_weight = region->points[i].r;
       // if (weight > obj_thresh) {
       // 	   graph->add_tweights(i, HARD_THRESH, 0);
       // } else
       if (/*weight < bkgd_thresh*/ edge_weight > 0) {
	   graph->add_tweights(i, 0, HARD_THRESH);
       } else if (!label_cache[i]){
	   float w = -std::log(weight/255.0) * 10;
	   if (weight == 0) {
	       w = -std::log(1e-9);
	   }
	   // if (isnan(w)) {
	   // }
	   // std::cout<< "\t" << w << "\t" << weight<< "\n";
	   graph->add_tweights(i, w, w);
       }
	
       for (int j = 0; j < this->neigbor_indices_[i].size(); j++) {
          int indx = this->neigbor_indices_[i][j];
          if (indx != i) {
	      // float w = std::pow(weight - weight_map->points[indx].r, 2);
	      float r = std::abs(cloud->points[indx].r - cloud->points[i].r);
	      float g = std::abs(cloud->points[indx].g - cloud->points[i].g);
	      float b = std::abs(cloud->points[indx].b - cloud->points[i].b);
	      float val = (r*r + g*g + b*b) /(255.0f * 255.0f);

	      if (val < 0.00001f) {
		  val = 0.00001f;
	      }
	      float w = fabs(std::log(val));
	      // if (w < 0.000000001) {
	      // 	  w = 0.000000001;
	      // }
	      

	      // w = std::sqrt(w);
	      // std::cout << w << "\t" << val  << "\n ";
	      graph->add_edge(i, indx, (w), (w));
          }
       }
    }
    
    
    ROS_INFO("\033[34m COMPUTING FLOW \033[0m");
    
    float flow = graph->maxflow();

    ROS_INFO("\033[34m FLOW: %3.2f \033[0m", flow);
    // plot
    region->clear();
    for (int i = 0; i < node_num; i++) {
       if (graph->what_segment(i) == GraphType::SOURCE) {
          region->push_back(cloud->points[i]);
       } else {
          continue;
       }
    }
    ROS_INFO("\033[34m DONE: %d \033[0m", region->size());

    cloud->clear();
    *cloud = *weight_map;
}
