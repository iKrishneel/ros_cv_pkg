
#include <collision_check_grasp_plannar/collision_check_grasp_plannar.h>

CollisionCheckGraspPlannar::CollisionCheckGraspPlannar() {
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    this->onInit();
}

void CollisionCheckGraspPlannar::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/collision_check_grasp_plannar/output/cloud", 1);
    
}

void CollisionCheckGraspPlannar::subscribe() {
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_boxes_.subscribe(this->pnh_, "input_boxes", 1);
    this->sub_indices_.subscribe(this->pnh_, "input_indices", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(this->sub_cloud_,
                              this->sub_indices_, this->sub_boxes_);
    this->sync_->registerCallback(
       boost::bind(&CollisionCheckGraspPlannar::cloudCB, this, _1, _2, _3));
}

void CollisionCheckGraspPlannar::unsubscribe() {
    this->sub_boxes_.unsubscribe();
    this->sub_cloud_.unsubscribe();
    this->sub_indices_.unsubscribe();
}

void CollisionCheckGraspPlannar::cloudCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const jsk_msgs::ClusterPointIndices::ConstPtr &indices_msg,
    const jsk_msgs::BoundingBoxArray::ConstPtr &boxes_msg) {
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    std::vector<IndicesMap> indices_map(static_cast<int>(cloud->size()));
    for (int i = 0; i < indices_msg->cluster_indices.size(); i++) {
       for (int j = 0; j < indices_msg->cluster_indices[
               i].indices.size(); j++) {
          IndicesMap im;
          im.index = indices_msg->cluster_indices[i].indices[j];
          im.label = i;
          indices_map[im.index] = im;
       }
    }

    this->kdtree_->setInputCloud(cloud);

    std::vector<std::vector<int> > adjacency_list(
       static_cast<int>(indices_msg->cluster_indices.size()));
    for (int i = 0; i < indices_msg->cluster_indices.size(); i++) {
       std::vector<int> label_maps;
       int prev_label = -1;
       for (int j = 0; j < indices_msg->cluster_indices[
               i].indices.size(); j++) {
          int indx = indices_msg->cluster_indices[i].indices[j];
          std::vector<int> neigbor_indices;
          this->getPointNeigbour<int>(neigbor_indices, cloud->points[indx]);
          for (int k = 1; k < neigbor_indices.size(); k++) {
             int index = neigbor_indices[k];
             if (indices_map[index].label != indices_map[indx].label &&
                 indices_map[index].label != prev_label) {
                label_maps.push_back(indices_map[index].label);
                prev_label = indices_map[index].label;
             }
          }
       }
       if (!label_maps.empty()) {
          std::sort(label_maps.begin(), label_maps.end(), sortVector);
          prev_label = -1;
          std::vector<int> adj_list;
          for (int k = 0; k < label_maps.size(); k++) {
             if (label_maps[k] != prev_label) {
                adj_list.push_back(label_maps[k]);
                prev_label = label_maps[k];
             }
          }
          adjacency_list[i] = adj_list;
       }
    }
}


void CollisionCheckGraspPlannar::getBoundingBoxGraspPoints(
    std::vector<Eigen::Vector3f> &grasp_points,
    const jsk_msgs::BoundingBox bounding_box) {
    Eigen::Vector3f center = Eigen::Vector3f(bounding_box.pose.position.x,
                                             bounding_box.pose.position.y,
                                             bounding_box.pose.position.z);
    Eigen::Vector3f dims = Eigen::Vector3f(bounding_box.dimensions.x / 2.0f,
                                           bounding_box.dimensions.y / 2.0f,
                                           bounding_box.dimensions.z / 2.0f);
    std::vector<Facets> side_points(NUMBER_OF_SIDE);
    Facets top;
    top.AA = Eigen::Vector3f(0.0f, -dims(1), dims(2));
    top.AB = Eigen::Vector3f(0.0f, dims(1), dims(2));
    top.BB = Eigen::Vector3f(-dims(0), 0.0f, dims(2));
    top.BA = Eigen::Vector3f(dims(0), 0.0f, dims(2));
    side_points[0] = top;
    
    Facets front;
    front.AA = Eigen::Vector3f(dims(0), 0.0f, dims(2));
    front.AB = Eigen::Vector3f(dims(0), 0.0f, -dims(2));
    front.BB = Eigen::Vector3f(dims(0), -dims(0), 0.0f);
    front.BA = Eigen::Vector3f(dims(0), dims(0), 0.0f);
    side_points[1] = front;
    
    Facets right;
    right.AA = Eigen::Vector3f(-dims(0), dims(1), 0.0f);
    right.AB = Eigen::Vector3f(dims(0), dims(1), 0.0f);
    right.BB = Eigen::Vector3f(0.0f, dims(1), -dims(2));
    right.BA = Eigen::Vector3f(0.0f, dims(1), dims(2));
    side_points[2] = right;
    
    Facets left;
    left.AA = Eigen::Vector3f(-dims(0), -dims(1), 0.0f);
    left.AB = Eigen::Vector3f(dims(0), -dims(1), 0.0f);
    left.BB = Eigen::Vector3f(0.0f, -dims(1), -dims(2));
    left.BA = Eigen::Vector3f(0.0f, -dims(1), dims(2));
    side_points[3] = left;
    
    PointCloud::Ptr box_points(new PointCloud);
    for (int i = 0; i < NUMBER_OF_SIDE; i++) {
       box_points->push_back(vector3f2PointT(side_points[i].AA));
       box_points->push_back(vector3f2PointT(side_points[i].AB));
       box_points->push_back(vector3f2PointT(side_points[i].BB));
       box_points->push_back(vector3f2PointT(side_points[i].BA));
    }

    Eigen::Quaternion<float> quaternion = Eigen::Quaternion<float>(
       bounding_box.pose.orientation.w, bounding_box.pose.orientation.x,
       bounding_box.pose.orientation.y, bounding_box.pose.orientation.z);
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() <<
       bounding_box.pose.position.x,
       bounding_box.pose.position.y,
       bounding_box.pose.position.z;
    transform.rotate(quaternion);
    pcl::transformPointCloud(*box_points, *box_points, transform);

}


template<class T>
void CollisionCheckGraspPlannar::getPointNeigbour(
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

CollisionCheckGraspPlannar::PointT
CollisionCheckGraspPlannar::vector3f2PointT(
    const Eigen::Vector3f vec, Eigen::Vector3f color) {
    PointT pt;
    pt.x = vec(0); pt.r = color(0);
    pt.y = vec(1); pt.g = color(1);
    pt.z = vec(2); pt.b = color(2);
    return pt;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "collision_check_grasp_plannar");
    CollisionCheckGraspPlannar ccgp;
    ros::spin();
    return 0;
}
