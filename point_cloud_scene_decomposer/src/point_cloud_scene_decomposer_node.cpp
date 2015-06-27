
#include <point_cloud_scene_decomposer/point_cloud_scene_decomposer.h>
#include <point_cloud_scene_decomposer/ClusterVoxels.h>

#include <vector>
#include <map>


PointCloudSceneDecomposer::PointCloudSceneDecomposer() :
    max_distance_(1.5f),
    start_signal_(true),
    processing_counter_(0),
    normal_(pcl::PointCloud<pcl::Normal>::Ptr(
                new pcl::PointCloud<pcl::Normal>)),
    orig_cloud_(pcl::PointCloud<PointT>::Ptr(
                        new pcl::PointCloud<PointT>)) {
    this->subscribe();
    this->onInit();
}

void PointCloudSceneDecomposer::onInit() {
    this->pub_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>(
        "/scene_decomposer/output/cloud_cluster", sizeof(char));
    this->pub_cloud_orig_ = nh_.advertise<sensor_msgs::PointCloud2>(
       "/scene_decomposer/output/original_cloud", sizeof(char));
    this->pub_indices_ = nh_.advertise<
          jsk_recognition_msgs::ClusterPointIndices>(
             "/scene_decomposer/output/indices", sizeof(char));
    this->pub_image_ = nh_.advertise<sensor_msgs::Image>(
        "/scene_decomposer/output/image", sizeof(char));
    this->pub_signal_ = nh_.advertise<point_cloud_scene_decomposer::signal>(
       "/scene_decomposer/output/signal", sizeof(char));
}

void PointCloudSceneDecomposer::subscribe() {

    this->sub_signal_ = nh_.subscribe(
       "input_signal", 1, &PointCloudSceneDecomposer::signalCallback, this);
    this->sub_cloud_ori_ = nh_.subscribe(
       "input_orig_cloud", 1,
       &PointCloudSceneDecomposer::origcloudCallback, this);
    this->sub_image_ = nh_.subscribe(
       "input_image", 1, &PointCloudSceneDecomposer::imageCallback, this);
    this->sub_norm_ = nh_.subscribe(
       "input_norm", 1, &PointCloudSceneDecomposer::normalCallback, this);
    this->sub_indices_ = nh_.subscribe(
       "input_indices", 1, &PointCloudSceneDecomposer::indicesCallback, this);
    this->sub_cloud_ = nh_.subscribe(
       "input_cloud", 1, &PointCloudSceneDecomposer::cloudCallback, this);
}

void PointCloudSceneDecomposer::unsubscribe() {
    this->sub_cloud_.shutdown();
    this->sub_norm_.shutdown();
    this->sub_image_.shutdown();
}

void PointCloudSceneDecomposer::signalCallback(
    const point_cloud_scene_decomposer::signal &signal_msg) {
    this->signal_ = signal_msg;
}

/**
 * subscriber to the moved objected marked as label
 */
void PointCloudSceneDecomposer::indicesCallback(
    const jsk_recognition_msgs::ClusterPointIndices & indices_msg) {
   /*   
    this->manipulated_obj_indices_.clear();
    for (int i = 0; i < indices_msg.cluster_indices.size(); i++) {
       pcl_msgs::PointIndices ros_msg = indices_msg.cluster_indices[i];
       pcl::PointIndices ind;
       ind.indices = ros_msg.indices;
       manipulated_obj_indices_.push_back(ind);
    }
   */
}

void PointCloudSceneDecomposer::origcloudCallback(
    const sensor_msgs::PointCloud2ConstPtr &orig_cloud_msg) {
    this->orig_cloud_ = pcl::PointCloud<PointT>::Ptr(
        new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*orig_cloud_msg, *orig_cloud_);
}

void PointCloudSceneDecomposer::imageCallback(
    const sensor_msgs::Image::ConstPtr &image_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(
           image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    this->image_ = cv_ptr->image.clone();
    std::cout << "Image: " << image_.size() << std::endl;
    // this->pub_image_.publish(cv_ptr->toImageMsg());
}

void PointCloudSceneDecomposer::normalCallback(
    const sensor_msgs::PointCloud2ConstPtr &normal_msg) {
    this->normal_ = pcl::PointCloud<pcl::Normal>::Ptr(
       new pcl::PointCloud<pcl::Normal>);
    pcl::fromROSMsg(*normal_msg, *normal_);
}

void PointCloudSceneDecomposer::cloudCallback(
    const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    std::cout << cloud->size() << "\t" << normal_->size()
              << "\t" <<  orig_cloud_->size() << image_.size() << std::endl;
    if (cloud->empty() || this->normal_->empty() || this->image_.empty() ||
        this->orig_cloud_->empty()) {
       ROS_ERROR("-- CANNOT PROCESS EMPTY INSTANCE");
       return;
    }

    /*
    jsk_recognition_msgs::BoundingBoxArray known_object_bboxes;
    if (!this->start_signal_) {
       this->getKnownObjectRegion(known_object_bboxes, 0.05f);
    }
    */
    
    if (this->start_signal_ ||
        ((this->processing_counter_ == this->signal_.counter) &&
         (this->signal_.command == 1))) {
       cv::Mat image = this->image_.clone();
       cv::Mat edge_map;
       this->getRGBEdge(image, edge_map, "cvCanny");
       // this->getDepthEdge(dep_img, edge_map, true);
    
       std::vector<cvPatch<int> > patch_label;
       this->cvLabelEgdeMap(orig_cloud_, image, edge_map, patch_label);

   
       pcl::PointCloud<PointT>::Ptr patch_cloud(new pcl::PointCloud<PointT>);
       pcl::copyPointCloud<PointT, PointT> (*orig_cloud_, *patch_cloud);

       std::vector<pcl::PointCloud<PointT>::Ptr> cloud_clusters;
       std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normal_clusters;

       pcl::PointCloud<pcl::PointXYZ>::Ptr centroids(
          new pcl::PointCloud<pcl::PointXYZ>);
       this->extractPointCloudClustersFrom2DMap(
          patch_cloud, patch_label, cloud_clusters,
          normal_clusters, centroids, image.size());


       const int neigbour_size = 4; /*5*/
       std::vector<std::vector<int> > neigbour_idx;
       this->pclNearestNeigborSearch(
          centroids, neigbour_idx, true, neigbour_size, 0.03);
       RegionAdjacencyGraph *rag = new RegionAdjacencyGraph();
       // const int edge_weight_criteria = rag->RAG_EDGE_WEIGHT_CONVEX_CRITERIA;
       const int edge_weight_criteria = rag->RAG_EDGE_WEIGHT_DISTANCE;
       rag->generateRAG(cloud_clusters, normal_clusters,
                        centroids, neigbour_idx, edge_weight_criteria);
       rag->splitMergeRAG(cloud_clusters, normal_clusters,
                          edge_weight_criteria, 0.50);
       std::vector<int> labelMD;
       rag->getCloudClusterLabels(labelMD);
       free(rag);
       std::vector<pcl::PointIndices> all_indices;
       this->semanticCloudLabel(
          cloud_clusters, cloud, labelMD, all_indices, rag->total_label);
       this->objectCloudClusterPostProcessing(cloud, all_indices);
       // --
       std::cout << YELLOW  << "-- NUMBER OF BOUNDING BOXES: "
                 << all_indices.size() << YELLOW  << RESET << std::endl;
       //--
       jsk_recognition_msgs::ClusterPointIndices ros_indices;
       ros_indices.cluster_indices = this->convertToROSPointIndices(
          all_indices, cloud_msg->header);
       ros_indices.header = cloud_msg->header;
    
       sensor_msgs::PointCloud2 ros_cloud;
       pcl::toROSMsg(*cloud, ros_cloud);
       ros_cloud.header = cloud_msg->header;

       // publishing the original subcribed cloud
       sensor_msgs::PointCloud2 ros_cloud_orig;
       pcl::toROSMsg(*orig_cloud_, ros_cloud_orig);
       ros_cloud_orig.header = cloud_msg->header;
       
       this->start_signal_ = false;  // turn off start up signal
       this->processing_counter_++;
       this->publishing_indices.cluster_indices.clear();
       this->publishing_cloud.data.clear();
       this->publishing_indices = ros_indices;
       this->publishing_cloud = ros_cloud;
       this->publishing_cloud_orig = ros_cloud_orig;

       image_msg = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
       image_msg->header = cloud_msg->header;
       image_msg->encoding = sensor_msgs::image_encodings::BGR8;
       image_msg->image = this->image_.clone();
       this->pub_image_.publish(image_msg->toImageMsg());
       
       this->pub_indices_.publish(ros_indices);
       this->pub_cloud_.publish(ros_cloud);
       this->pub_cloud_orig_.publish(ros_cloud_orig);
    } else {
       ROS_WARN("-- PUBLISHING OLD DATA.");
       if (this->processing_counter_ != 0) {
          this->publishing_indices.header = cloud_msg->header;
          this->publishing_cloud.header = cloud_msg->header;
          this->publishing_cloud_orig.header = cloud_msg->header;
          this->pub_indices_.publish(this->publishing_indices);
          this->pub_cloud_.publish(this->publishing_cloud);
          this->pub_cloud_orig_.publish(this->publishing_cloud_orig);
          image_msg->header = cloud_msg->header;
          this->pub_image_.publish(image_msg->toImageMsg());
       }
    }
    point_cloud_scene_decomposer::signal pub_sig;
    pub_sig.header = cloud_msg->header;
    pub_sig.command = 2;
    pub_sig.counter = this->processing_counter_ - 1;
    this->pub_signal_.publish(pub_sig);

    std::cout << "Processing Counter: " << pub_sig.command
              << "\t Signal: " << pub_sig.counter << std::endl;
}


void PointCloudSceneDecomposer::pclNearestNeigborSearch(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    std::vector<std::vector<int> > &pointIndices,
    bool isneigbour,
    const int k,
    const double radius) {
    if (cloud->empty()) {
       ROS_ERROR("Cannot search NN in an empty point cloud");
       return;
    }
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);
    std::vector<std::vector<float> > pointSquaredDistance;
    for (int i = 0; i < cloud->size(); i++) {
       std::vector<int>pointIdx;
       std::vector<float> pointSqDist;
       pcl::PointXYZ searchPoint = cloud->points[i];
       if (isneigbour) {
          kdtree.nearestKSearch(searchPoint, k, pointIdx, pointSqDist);
       } else {
          kdtree.radiusSearch(searchPoint, radius, pointIdx, pointSqDist);
       }
       pointIndices.push_back(pointIdx);
       pointSquaredDistance.push_back(pointSqDist);
       pointIdx.clear();
       pointSqDist.clear();
    }
}

void PointCloudSceneDecomposer::extractPointCloudClustersFrom2DMap(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const std::vector<cvPatch<int> > &patch_label,
    std::vector<pcl::PointCloud<PointT>::Ptr> &cloud_clusters,
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> &normal_clusters,
    pcl::PointCloud<pcl::PointXYZ>::Ptr centroids,
    const cv::Size isize) {
    if (cloud->empty() || patch_label.empty()) {
       ROS_ERROR("ERROR: Point Cloud vector is empty...");
       return;
    }
    const int FILTER_SIZE = 10;
    int icounter = 0;
    cloud_clusters.clear();
    pcl::ExtractIndices<PointT>::Ptr eifilter(
       new pcl::ExtractIndices<PointT>);
    eifilter->setInputCloud(cloud);
    pcl::ExtractIndices<pcl::Normal>::Ptr n_eifilter(
       new pcl::ExtractIndices<pcl::Normal>);
    n_eifilter->setInputCloud(this->normal_);
    
    for (int k = 0; k < patch_label.size(); k++) {
       std::vector<std::vector<int> > cluster_indices(
           static_cast<int>(100));  // CHANGE TO AUTO-SIZE
       cv::Mat labelMD = patch_label[k].patch.clone();
       cv::Rect_<int> rect = patch_label[k].rect;
       if (patch_label[k].is_region) {
          for (std::vector<cv::Point2i>::const_iterator it =
                  patch_label[k].region.begin();
               it != patch_label[k].region.end(); it++) {
             int x = it->x + rect.x;
             int y = it->y + rect.y;
             int label_ = static_cast<int>(labelMD.at<float>(it->y, it->x));
             int index = (x + (y * isize.width));
             if (cloud->points[index].z <= this->max_distance_) {
                cluster_indices[label_].push_back(index);
             }
          }
          for (int i = 0; i < cluster_indices.size(); i++) {
             pcl::PointCloud<PointT>::Ptr tmp_cloud(
                new pcl::PointCloud<PointT>);
             pcl::PointIndices::Ptr indices(
                new pcl::PointIndices());
             indices->indices = cluster_indices[i];
             eifilter->setIndices(indices);
             eifilter->filter(*tmp_cloud);
             // filter the normal
             pcl::PointCloud<pcl::Normal>::Ptr tmp_normal(
                new pcl::PointCloud<pcl::Normal>);
             n_eifilter->setIndices(indices);
             n_eifilter->filter(*tmp_normal);
             if (tmp_cloud->width > FILTER_SIZE) {
                Eigen::Vector4f centroid;
                pcl::compute3DCentroid<PointT, float>(
                   *cloud, *indices, centroid);
                float ct_x = static_cast<float>(centroid[0]);
                float ct_y = static_cast<float>(centroid[1]);
                float ct_z = static_cast<float>(centroid[2]);
                if (!isnan(ct_x) && !isnan(ct_y) && !isnan(ct_z)) {
                   centroids->push_back(pcl::PointXYZ(ct_x, ct_y, ct_z));
                   cloud_clusters.push_back(tmp_cloud);
                   normal_clusters.push_back(tmp_normal);
                }
             }
          }
       }
       cluster_indices.clear();
    }
}

void PointCloudSceneDecomposer::semanticCloudLabel(
    const std::vector<pcl::PointCloud<PointT>::Ptr> &cloud_clusters,
    pcl::PointCloud<PointT>::Ptr cloud,
    const std::vector<int> &labelMD,
    std::vector<pcl::PointIndices> &all_indices,
    const int total_label) {
    cloud->clear();
    all_indices.clear();
    all_indices.resize(total_label);
    int icounter = 0;
    for (int i = 0; i < cloud_clusters.size(); i++) {
       int _idx = labelMD.at(i);
       for (int j = 0; j < cloud_clusters[i]->size(); j++) {
          PointT pt;
          pt.x = cloud_clusters[i]->points[j].x;
          pt.y = cloud_clusters[i]->points[j].y;
          pt.z = cloud_clusters[i]->points[j].z;
          pt.r = this->color[_idx].val[0];
          pt.g = this->color[_idx].val[1];
          pt.b = this->color[_idx].val[2];
          cloud->push_back(pt);
          all_indices[_idx].indices.push_back(icounter++);
       }
    }
}

/**
 * function to divide 2 voxels of same label but not connected in 3D space
 */
void PointCloudSceneDecomposer::objectCloudClusterPostProcessing(
    pcl::PointCloud<PointT>::Ptr in_cloud,
    std::vector<pcl::PointIndices> &all_indices,
    const int min_cluster_size) {
    if (in_cloud->empty()) {
       ROS_ERROR("-- NOT CLUSTER TO PROCESS");
       return;
    }
    std::vector<pcl::PointIndices> clustered_indices;
    clustered_indices.clear();
    
    pcl::search::KdTree<PointT>::Ptr tree(
       new pcl::search::KdTree<PointT>);
    tree->setInputCloud(in_cloud);

    for (std::vector<pcl::PointIndices>::iterator it =
            all_indices.begin(); it != all_indices.end(); it++) {
       pcl::PointIndices::Ptr indices(new pcl::PointIndices());
       indices->indices = it->indices;
       std::vector<pcl::PointIndices> cluster_indices;
       pcl::EuclideanClusterExtraction<PointT> euclidean_clustering;
       euclidean_clustering.setClusterTolerance(0.02);
       euclidean_clustering.setMinClusterSize(min_cluster_size);
       euclidean_clustering.setMaxClusterSize(25000);
       euclidean_clustering.setSearchMethod(tree);
       euclidean_clustering.setInputCloud(in_cloud);
       euclidean_clustering.setIndices(indices);
       euclidean_clustering.extract(cluster_indices);

       clustered_indices.insert(clustered_indices.end(),
                                cluster_indices.begin(),
                                cluster_indices.end());
    }
    all_indices.clear();
    all_indices.insert(all_indices.end(),
                       clustered_indices.begin(),
                       clustered_indices.end());
}

std::vector<pcl_msgs::PointIndices>
PointCloudSceneDecomposer::convertToROSPointIndices(
    const std::vector<pcl::PointIndices> cluster_indices,
    const std_msgs::Header& header) {
    std::vector<pcl_msgs::PointIndices> ret;
    for (size_t i = 0; i < cluster_indices.size(); i++) {
       pcl_msgs::PointIndices ros_msg;
       ros_msg.header = header;
       ros_msg.indices = cluster_indices[i].indices;
       ret.push_back(ros_msg);
    }
    return ret;
}

int main(int argc, char *argv[]) {

    ros::init(argc, argv, "point_cloud_scene_decomposer");
    srand(time(NULL));
    PointCloudSceneDecomposer pcsd;
    ros::spin();
    return 0;
}
