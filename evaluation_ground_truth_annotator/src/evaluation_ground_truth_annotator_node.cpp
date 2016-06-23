
#include <evaluation_ground_truth_annotator/evaluation_ground_truth_annotator.h>

CvAlgorithmEvaluation::CvAlgorithmEvaluation() :
    labels_(0) {
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    this->marked_cloud_ = PointCloud::Ptr(new PointCloud);
    this->all_indices_.clear();
    this->pnh_.getParam("pcd_path", this->save_path_);

    this->save_path_ = "/home/krishneel/Desktop/acvr-eccv/ground-truth/scene3.pcd";
    std::string bag_path = "/home/krishneel/Desktop/acvr-eccv/scene5.bag";

    this->onInit();

    // this->readBagFile(bag_path);
}

void CvAlgorithmEvaluation::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/output/cloud", 1);
    this->pub_indices_ = this->pnh_.advertise<jsk_msgs::ClusterPointIndices>(
       "/output/indices", 1);
}

void CvAlgorithmEvaluation::subscribe() {
    this->sub_gt_cloud_ = this->pnh_.subscribe(
       "input_gt_cloud", 1, &CvAlgorithmEvaluation::groundTCB, this);

   
    this->sub_cloud_.subscribe(this->pnh_, "input_cloud", 1);
    this->sub_indices_.subscribe(this->pnh_, "input_indices", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(this->sub_cloud_,
                              this->sub_indices_);
    this->sync_->registerCallback(
       boost::bind(&CvAlgorithmEvaluation::cloudCB, this, _1, _2));
}

void CvAlgorithmEvaluation::unsubscribe() {
   
}


void CvAlgorithmEvaluation::readBagFile(
    const std::string bag_path) {

    ROS_WARN("READING BAG");
   
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);

    std::string cloud_topic = std::string("/cbss/output/cloud");
    std::string indices_topic = std::string("/cbss/output/indices");
    
    std::vector<std::string> topics;
    topics.push_back(cloud_topic);
    topics.push_back(indices_topic);

    rosbag::View view(bag, rosbag::TopicQuery(topics));
    
    int icounter = 0;
    foreach(rosbag::MessageInstance const m, view) {
       sensor_msgs::PointCloud2::ConstPtr cloud_msg(
          new sensor_msgs::PointCloud2);
       jsk_msgs::ClusterPointIndices::ConstPtr indices_msg(
          new jsk_msgs::ClusterPointIndices);
       if (m.getTopic() == cloud_topic) {
          cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
       }
       if (m.getTopic() == indices_topic) {
          indices_msg = m.instantiate<jsk_msgs::ClusterPointIndices>();
       }
       if (cloud_msg != NULL && indices_msg != NULL) {
          this->cloudCB(cloud_msg, indices_msg);
       }
    }
    bag.close();
}


void CvAlgorithmEvaluation::cloudCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const jsk_msgs::ClusterPointIndices::ConstPtr &indices_msg) {
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if (this->save_path_.empty()) {
       ROS_ERROR("NO PCD PATH FOUND");
       return;
    }

    pcl::io::savePCDFileASCII("cloud.pcd", *cloud);
    
    PointCloud::Ptr ground_truth(new PointCloud);
    int read = pcl::io::loadPCDFile<PointT> (this->save_path_, *ground_truth);
    if (read == -1) {
       ROS_ERROR("NO PCD NOT READ");
       return;
    }

    int prev_index = 0;
    std::vector<pcl::PointIndices> all_indices;
    pcl::PointIndices indices;
    for (int i = 0; i < ground_truth->size(); i++) {
       PointT pt = ground_truth->points[i];
       if (pt.r != prev_index) {
          all_indices.push_back(indices);
          indices.indices.clear();
          prev_index++;
       }
       indices.indices.push_back(i);
    }

    std::vector<pcl::PointIndices> object_indices;
    for (int i = 0; i < indices_msg->cluster_indices.size(); i++) {
       pcl::PointIndices o_indices;
       for (int j = 0; j < indices_msg->cluster_indices[
               i].indices.size(); j++) {
          int idx = indices_msg->cluster_indices[i].indices[j];
          o_indices.indices.push_back(idx);
       }
       object_indices.push_back(o_indices);
    }

    // std::cout << "INFO: " << object_indices.size()  << "\n";
    // std::cout << "INFO: " << all_indices.size()  << "\n";
    // std::cout << "INFO: " << cloud->size()  << "\n";
    
    this->intersectionUnionCoef(cloud, ground_truth,
                                object_indices, all_indices);
    
    /*
    jsk_msgs::ClusterPointIndices ros_indices;
    ros_indices.cluster_indices = pcl_conversions::convertToROSPointIndices(
       all_indices, cloud_msg->header);
    ros_indices.header = cloud_msg->header;
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*ground_truth, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_indices_.publish(ros_indices);
    this->pub_cloud_.publish(ros_cloud);
    */
}

void CvAlgorithmEvaluation::intersectionUnionCoef(
    const PointCloud::Ptr cloud, const PointCloud::Ptr gt_cloud,
    const std::vector<pcl::PointIndices> all_indices,
    const std::vector<pcl::PointIndices> gt_all_indices) {
    if (cloud->empty() || gt_cloud->empty() ||
        all_indices.empty() || gt_all_indices.empty()) {
       ROS_ERROR("EMPTY DATA");
       return;
    }

    std::vector<PointCloud::Ptr> object_cloud;
    std::vector<Eigen::Vector4f> object_centroid;
    for (int i = 0; i < all_indices.size(); i++) {
       PointCloud::Ptr region(new PointCloud);
       for (int j = 0; j < all_indices[i].indices.size() ; j++) {
          int idx = all_indices[i].indices[j];
          region->push_back(cloud->points[idx]);
       }
       Eigen::Vector4f centroid;
       pcl::compute3DCentroid<PointT, float>(*region, centroid);
       centroid(3) = 1.0f;
       object_centroid.push_back(centroid);
       object_cloud.push_back(region);
    }
       
    for (int i = 0; i < gt_all_indices.size(); i++) {
       PointCloud::Ptr region(new PointCloud);
       for (int j = 0; j < gt_all_indices[i].indices.size() ; j++) {
          int idx = gt_all_indices[i].indices[j];
          region->push_back(gt_cloud->points[idx]);
       }
       this->kdtree_->setInputCloud(region);

       Eigen::Vector4f centroid;
       pcl::compute3DCentroid<PointT, float>(*region, centroid);
       centroid(3) = 1.0f;
       
       double dist = DBL_MAX;
       int c_index = -1;
       for (int k = 0; k < object_centroid.size(); k++) {
          double d = pcl::distances::l2(centroid, object_centroid[k]);
          if (d < dist) {
             dist = d;
             c_index = k;
          }
       }

       // std::cout << "\033[31m index: \033[0m"  << c_index << "\n";
       
       if (c_index != -1) {
          int match_counter = 0;
          for (int k = 0; k < object_cloud[c_index]->size(); k++) {
             PointT ppt = object_cloud[c_index]->points[k];
             std::vector<int> neigbor_indices;
             this->getPointNeigbour<int>(neigbor_indices, ppt, 1);
             Eigen::Vector4f gpt = region->points[
                neigbor_indices[0]].getVector4fMap();
             gpt(3) = 1.0f;
             Eigen::Vector4f cpt = ppt.getVector4fMap();
             cpt(3) = 1.0f;
             double d = pcl::distances::l2(cpt, gpt);
             // std::cout << "distance: " << d  << "\n";
             if (d < 0.01) {
                match_counter++;
             }
          }


          float acc = static_cast<float>(match_counter)/
             static_cast<float>(region->size());

          ROS_WARN("ACCUARCY: %3.2f ", acc);
          
          std::cout << "Match: " << match_counter << "\t" <<
             object_cloud[c_index]->size() << "\t"
                    << region->size()  << "\n";

       }
    }
    std::cout   << "\n";

}

template<class T>
void CvAlgorithmEvaluation::getPointNeigbour(
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


void CvAlgorithmEvaluation::groundTCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    ROS_WARN("RECEIVED GROUND TRUTH CLIUD: %d", cloud->size());

    for (int i = 0; i < cloud->size(); i++) {
       PointT pt = cloud->points[i];
       pt.r = this->labels_;
       pt.b = this->labels_;
       pt.g = this->labels_;
       this->marked_cloud_->push_back(pt);
    }
    
    char ch;
    std::cout << "pressed y to save:"  << "\n";
    std::cin >> ch;
    if (ch == 'y') {
       ROS_ERROR("saving the marked cloud: %d", cloud->size());
       std::string save_path = "/home/krishneel/Desktop/acvr-eccv/ground-truth/";
       std::string name = "scene8.pcd";

       pcl::io::savePCDFileASCII(save_path + name, *marked_cloud_);
       this->marked_cloud_ = PointCloud::Ptr(new PointCloud);
       this->labels_ = 0;
       
    } else {
       this->labels_++;
    }
    std::cout << "label: " << labels_  << "\n";
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "evaluation_ground_truth_annotator");
    CvAlgorithmEvaluation cvae;
    ros::spin();
    return 0;
}
