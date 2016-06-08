
#include <collision_check_grasp_plannar/collision_check_grasp_plannar.h>

CollisionCheckGraspPlannar::CollisionCheckGraspPlannar() :
    search_radius_thresh_(0.04) {
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    this->onInit();
}

void CollisionCheckGraspPlannar::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/collision_check_grasp_plannar/output/cloud", 1);
    this->pub_bbox_ = this->pnh_.advertise<jsk_msgs::BoundingBoxArray>(
       "/collision_check_grasp_plannar/output/grasp_boxes", 1);
    this->pub_grasp_ = this->pnh_.advertise<geometry_msgs::PoseArray>(
       "/collision_check_grasp_plannar/output/grasp_points", 1);
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

    this->header_ = boxes_msg->header;

    std::cout << "INDICES: " << indices_msg->cluster_indices.size()  << "\n";
    std::cout << "BOXES: " << boxes_msg->boxes.size()  << "\n\n";
    
    
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


    /**
     * TODO: first build adjacency_list and determine the support
     * structure, then get grasp points
     */
    
    std::vector<PointCloud::Ptr> all_grasp_points(
       static_cast<int>(boxes_msg->boxes.size()));
    std::vector<std::vector<bool> > flag_grasp_points(
       static_cast<int>(boxes_msg->boxes.size()));
    
    for (int i = 0; i < boxes_msg->boxes.size(); i++) {
       PointCloud::Ptr grasp_points(new PointCloud);
       this->getBoundingBoxGraspPoints(grasp_points, boxes_msg->boxes[i]);
       all_grasp_points[i] = grasp_points;
       for (int k = 0; k < grasp_points->size() - 1; k++) {
          flag_grasp_points[i].push_back(true);
       }
    }
    
    for (int i = 0; i < all_grasp_points.size(); i++) {
       PointCloud::Ptr grasp_points(new PointCloud);
       grasp_points = all_grasp_points[i];
       int csize = grasp_points->size() - 1;
       Eigen::Vector4f centroid = grasp_points->points[csize].getVector4fMap();
       centroid(3) = 1.0f;
       for (int j = 0; j < csize; j++) {
          Eigen::Vector4f g_point = grasp_points->points[i].getVector4fMap();
          g_point(3) = 1.0f;
          std::vector<int> neigbor_indices;
          this->getPointNeigbour<float>(neigbor_indices,
                                        grasp_points->points[j],
                                        this->search_radius_thresh_, false);
          int prev_label = -1;
          std::vector<int> label_maps;
          for (int k = 0; k < neigbor_indices.size(); k++) {
             int index = neigbor_indices[k];
             if (indices_map[index].label != i &&
                indices_map[index].label != prev_label) {
                label_maps.push_back(indices_map[index].label);
                prev_label = indices_map[index].label;
             }
          }
          if (label_maps.empty()) {
             flag_grasp_points[i][j] = true;
          } else {
             std::sort(label_maps.begin(), label_maps.end(), sortVector);
             prev_label = -1;
             for (int k = 0; k < label_maps.size(); k++) {
                if (label_maps[k] != prev_label) {
                   jsk_msgs::BoundingBox adj_box = boxes_msg->boxes[
                      label_maps[k]];
                   Eigen::Vector4f adj_centroid = Eigen::Vector4f(
                      adj_box.pose.position.x, adj_box.pose.position.y,
                      adj_box.pose.position.z, 1.0f);
                   double d_cent = pcl::distances::l2(centroid, adj_centroid);
                   double d_pt = pcl::distances::l2(g_point, adj_centroid);
                   if (d_pt > d_cent && flag_grasp_points[i][j]) {
                      flag_grasp_points[i][j] = true;
                   } else if (d_pt < d_cent && flag_grasp_points[i][j]) {
                      flag_grasp_points[i][j] = false;
                      break;
                   }
                   prev_label = label_maps[k];
                }
             }
          }
       }
    }

    jsk_msgs::BoundingBoxArray bbox_array;
    geometry_msgs::PoseArray grasp_pose;
    PointCloud::Ptr grasp_points(new PointCloud);
    for (int i = 0; i < flag_grasp_points.size(); i++) {
       std::vector<int> adjacency_list;
       if (!flag_grasp_points[i].empty()) {
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
             for (int k = 0; k < label_maps.size(); k++) {
                if (label_maps[k] != prev_label) {
                   adjacency_list.push_back(label_maps[k]);
                   prev_label = label_maps[k];
                }
             }
          }
       }
       bool good_bbox = false;
       if (adjacency_list.empty()) {
          for (int j = 0; j < flag_grasp_points[i].size(); j += 2) {
             if (flag_grasp_points[i][j] && flag_grasp_points[i][j+1]) {
                grasp_points->push_back(all_grasp_points[i]->points[j]);
                grasp_points->push_back(all_grasp_points[i]->points[j+1]);
                
                //! move points to center
                PointT pt;
                pointCenter(pt, all_grasp_points[i]->points[j],
                            all_grasp_points[i]->points[j+1]);
                pt.r = 0; pt.g = 255; pt.b = 0;
                grasp_points->push_back(pt);

                geometry_msgs::Pose gpose;
                gpose.position.x = pt.x;
                gpose.position.y = pt.y;
                gpose.position.z = pt.z;
                gpose.orientation = boxes_msg->boxes[i].pose.orientation;
                grasp_pose.poses.push_back(gpose);
             }
          }
          good_bbox = true;
       } else {
          bool is_grasp_pts = true;
          int g_size = all_grasp_points[i]->size() - 1;
          float center_y = all_grasp_points[i]->points[g_size].y;
          for (int j = 0; j < adjacency_list.size(); j++) {
             int idx = adjacency_list[j];
             g_size = all_grasp_points[idx]->size() - 1;
             float ncenter_y = all_grasp_points[idx]->points[g_size].y;
             if (ncenter_y < center_y && ncenter_y < center_y+(
                    boxes_msg->boxes[i].dimensions.y / 2.0f) + 0.005f) {
                is_grasp_pts = false;
                good_bbox = false;
             }
          }
          if (is_grasp_pts) {
             good_bbox = true;
             for (int j = 0; j < flag_grasp_points[i].size(); j += 2) {
                if (flag_grasp_points[i][j] && flag_grasp_points[i][j+1]) {
                   grasp_points->push_back(all_grasp_points[i]->points[j]);
                   grasp_points->push_back(all_grasp_points[i]->points[j+1]);

                   PointT pt;
                   pointCenter(pt, all_grasp_points[i]->points[j],
                               all_grasp_points[i]->points[j+1]);
                   pt.r = 0; pt.g = 255; pt.b = 0;
                   grasp_points->push_back(pt);
                   
                   geometry_msgs::Pose gpose;
                   gpose.position.x = pt.x;
                   gpose.position.y = pt.y;
                   gpose.position.z = pt.z;
                   gpose.orientation = boxes_msg->boxes[i].pose.orientation;
                   grasp_pose.poses.push_back(gpose);
                }
             }
          }
       }
       if (good_bbox) {
          bbox_array.boxes.push_back(boxes_msg->boxes[i]);
       }
    }

    bbox_array.header = boxes_msg->header;
    this->pub_bbox_.publish(bbox_array);

    grasp_pose.header = boxes_msg->header;
    this->pub_grasp_.publish(grasp_pose);
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*grasp_points, ros_cloud);
    ros_cloud.header = boxes_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}


void CollisionCheckGraspPlannar::getBoundingBoxGraspPoints(
    PointCloud::Ptr box_points, const jsk_msgs::BoundingBox bounding_box) {
    Eigen::Vector3f center = Eigen::Vector3f(bounding_box.pose.position.x,
                                             bounding_box.pose.position.y,
                                             bounding_box.pose.position.z);
    Eigen::Vector3f dims = Eigen::Vector3f(bounding_box.dimensions.x / 2.0f,
                                           bounding_box.dimensions.y / 2.0f,
                                           bounding_box.dimensions.z / 2.0f);
    std::vector<Facets> side_points(NUMBER_OF_SIDE);
    Facets top;
    top.AA = Eigen::Vector3f(0.0f, -dims(1), -dims(2));
    top.AB = Eigen::Vector3f(0.0f, dims(1), -dims(2));
    top.BB = Eigen::Vector3f(-dims(0), 0.0f, -dims(2));
    top.BA = Eigen::Vector3f(dims(0), 0.0f, -dims(2));
    side_points[0] = top;
    
    Facets front;
    front.AA = Eigen::Vector3f(-dims(0), dims(0), 0.0f);
    front.AB = Eigen::Vector3f(dims(0), dims(0), 0.0f);
    side_points[1] = front;
    
    Facets right;
    right.AA = Eigen::Vector3f(dims(0), -dims(1), 0.0f);
    right.AB = Eigen::Vector3f(dims(0), dims(1), 0.0f);
    side_points[2] = right;
    
    Facets left;
    left.AA = Eigen::Vector3f(-dims(0), dims(1), 0.0f);
    left.AB = Eigen::Vector3f(-dims(0), -dims(1), 0.0f);
    side_points[3] = left;

    Eigen::Vector3f color[NUMBER_OF_SIDE];
    color[0] = Eigen::Vector3f(255, 0, 0);
    color[1] = Eigen::Vector3f(0, 0, 255);
    color[2] = Eigen::Vector3f(255, 0, 255);
    color[3] = Eigen::Vector3f(255, 255, 22);
        
    
    for (int i = 0; i < side_points.size(); i++) {
       box_points->push_back(vector3f2PointT(side_points[i].AA, color[i]));
       box_points->push_back(vector3f2PointT(side_points[i].AB, color[i]));
       if (i == 0) {
          box_points->push_back(vector3f2PointT(side_points[i].BB, color[i]));
          box_points->push_back(vector3f2PointT(side_points[i].BA, color[i]));
       }
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
    box_points->push_back(vector3f2PointT(center));
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

void CollisionCheckGraspPlannar::pointCenter(
    PointT &point, const PointT pt1, const PointT pt2) {
    point.x = (pt1.x + pt2.x) / 2.0f;
    point.y = (pt1.y + pt2.y) / 2.0f;
    point.z = (pt1.z + pt2.z) / 2.0f;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "collision_check_grasp_plannar");
    CollisionCheckGraspPlannar ccgp;
    ros::spin();
    return 0;
}
