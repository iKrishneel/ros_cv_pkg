
#include <point_cloud_scene_decomposer/point_cloud_scene_decomposer.h>
#include <point_cloud_scene_decomposer/ClusterVoxels.h>

#include <vector>
#include <map>


PointCloudSceneDecomposer::PointCloudSceneDecomposer() :
    max_distance_(1.0f),
    normal_(pcl::PointCloud<pcl::Normal>::Ptr(
                new pcl::PointCloud<pcl::Normal>)),
    orig_cloud_(pcl::PointCloud<PointT>::Ptr(
                        new pcl::PointCloud<PointT>)) {
    this->srv_client_ = nh_.serviceClient<
       point_cloud_scene_decomposer::ClusterVoxels>("cluster_voxels");
    this->subscribe();
    this->onInit();
}

void PointCloudSceneDecomposer::onInit() {
    this->pub_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>(
        "/scene_decomposer/output/cloud", sizeof(char));
    this->pub_image_ = nh_.advertise<sensor_msgs::Image>(
        "/scene_decomposer/output/image", sizeof(char));
}

void PointCloudSceneDecomposer::subscribe() {

    this->sub_cloud_ori_ = nh_.subscribe("/camera/depth_registered/points", 1,
        &PointCloudSceneDecomposer::origcloudCallback , this);
    
    this->sub_image_ = nh_.subscribe("input_image", 1,
        &PointCloudSceneDecomposer::imageCallback, this);
    this->sub_norm_ = nh_.subscribe("input_norm", 1,
       &PointCloudSceneDecomposer::normalCallback, this);
    this->sub_cloud_ = nh_.subscribe("input_cloud", 1,
        &PointCloudSceneDecomposer::cloudCallback, this);
}

void PointCloudSceneDecomposer::unsubscribe() {
    this->sub_cloud_.shutdown();
    this->sub_norm_.shutdown();
    this->sub_image_.shutdown();
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
              << image_.size() << std::endl;
    if (cloud->empty() || this->normal_->empty() || this->image_.empty()) {
       ROS_ERROR("-- CANNOT PROCESS EMPTY INSTANCE");
       return;
    }
        
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
    this->extractPointCloudClustersFrom2DMap(patch_cloud, patch_label,
       cloud_clusters, normal_clusters, centroids, image.size());


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
    this->semanticCloudLabel(cloud_clusters, cloud, labelMD);
    
    cv::waitKey(3);
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
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


/**
 * compute the locate point gradient orientation
 */
void PointCloudSceneDecomposer::pointCloudLocalGradient(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr cloud_normal,
    cv::Mat &depth_img) {
    if (cloud->empty() || cloud_normal->empty() || depth_img.empty()) {
        ROS_ERROR("ERROR: Point Cloud Empty...");
        return;
    }
    if (cloud->width != depth_img.cols ||
        cloud_normal->width != depth_img.cols) {
        ROS_ERROR("ERROR: Incorrect size...");
        return;
    }
    const int start_pt = 1;
    cv::Mat normalMap = cv::Mat::zeros(depth_img.size(), CV_8UC3);
    cv::Mat localGradient = cv::Mat::zeros(depth_img.size(), CV_32F);
    for (int j = start_pt; j < depth_img.rows - start_pt; j++) {
       for (int i = start_pt; i < depth_img.cols - start_pt; i++) {
          int pt_index = i + (j * depth_img.cols);
          Eigen::Vector3f centerPointVec = Eigen::Vector3f(
             cloud_normal->points[pt_index].normal_x,
             cloud_normal->points[pt_index].normal_y,
             cloud_normal->points[pt_index].normal_z);
          int icounter = 0;
          float scalarProduct = 0.0f;
          for (int y = -start_pt; y <= start_pt; y++) {
             for (int x = -start_pt; x <= start_pt; x++) {
                if (x != 0 && y != 0) {
                   int n_index = (i + x) + ((j + y) * depth_img.cols);
                   Eigen::Vector3f neigbourPointVec = Eigen::Vector3f(
                      cloud_normal->points[n_index].normal_x,
                      cloud_normal->points[n_index].normal_y,
                      cloud_normal->points[n_index].normal_z);
                   scalarProduct += (neigbourPointVec.dot(centerPointVec) /
                                     (neigbourPointVec.norm() *
                                      centerPointVec.norm()));
                   ++icounter;
                }
             }
          }
          scalarProduct /= static_cast<float>(icounter);
          localGradient.at<float>(j, i) = static_cast<float>(scalarProduct);
          
          scalarProduct = 1 - scalarProduct;
          cv::Scalar jmap = JetColour<float, float, float>(
             scalarProduct, 0, 1);
          normalMap.at<cv::Vec3b>(j, i)[0] = jmap.val[0] * 255;
          normalMap.at<cv::Vec3b>(j, i)[1] = jmap.val[1] * 255;
          normalMap.at<cv::Vec3b>(j, i)[2] = jmap.val[2] * 255;
       }
    }
    cv::imshow("Local Gradient", normalMap);
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
    const std::vector<int> &labelMD) {
    cloud->clear();
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
       }
    }
}



void PointCloudSceneDecomposer::objectCloudClusterPostProcessing(
    std::vector<pcl::PointCloud<PointT>::Ptr> &cloud_clusters,
    std::vector<int> &labelMD,
    const int total_label,
    const int min_cluster_size) {
    if (cloud_clusters.size() != labelMD.size()) {
       ROS_ERROR("-- CLUSTER LABEL NOT SAME SIAZE");
       return;
    }
    std::vector<pcl::PointCloud<PointT>::Ptr> c_clusters;
    std::vector<int> c_labels;
    int icounter = 0;
    int last_label = total_label;
    for (std::vector<pcl::PointCloud<PointT>::Ptr>::const_iterator it =
            cloud_clusters.begin(); it != cloud_clusters.end(); it++) {
       if ((*it)->size() > min_cluster_size) {
          pcl::PointCloud<PointT>::Ptr cloud(
             new pcl::PointCloud<PointT>);
          pcl::copyPointCloud<PointT, PointT>(*(*it), *cloud);
          pcl::search::KdTree<PointT>::Ptr tree(
             new pcl::search::KdTree<PointT>);
          tree->setInputCloud(cloud);
          std::vector<pcl::PointIndices> cluster_indices;
          pcl::EuclideanClusterExtraction<PointT> euclidean_clustering;
          euclidean_clustering.setClusterTolerance(0.1);
          euclidean_clustering.setMinClusterSize(min_cluster_size);
          euclidean_clustering.setMaxClusterSize(25000);
          euclidean_clustering.setSearchMethod(tree);
          euclidean_clustering.setInputCloud(cloud);
          euclidean_clustering.extract(cluster_indices);
          int clabel = labelMD[icounter++];
          if (cluster_indices.size() > sizeof(char)) {
             int ccount = 1;
             for (std::vector<pcl::PointIndices>::const_iterator it =
                     cluster_indices.begin(); it != cluster_indices.end();
                  ++it) {
              pcl::PointCloud<PointT>::Ptr cloud_cluster(
                 new pcl::PointCloud<PointT>);
              for (std::vector<int>::const_iterator pit = it->indices.begin();
                   pit != it->indices.end(); ++pit) {
                 cloud_cluster->points.push_back(cloud->points[*pit]);
              }
              cloud_cluster->width = cloud_cluster->points.size();
              cloud_cluster->height = sizeof(char);
              cloud_cluster->is_dense = true;
              c_clusters.push_back(cloud_cluster);
              if (ccount++ == 1) {
                 c_labels.push_back(clabel);
              } else {
                 c_labels.push_back(last_label++);
              }
             }
          } else if (cluster_indices.size() == sizeof(char)) {
             c_clusters.push_back(cloud);
             c_labels.push_back(clabel);
          }
       }
    }
    cloud_clusters.clear();
    labelMD.clear();
    cloud_clusters.insert(
       cloud_clusters.end(), c_clusters.begin(), c_clusters.end());
    labelMD.insert(labelMD.end(), c_labels.begin(), c_labels.end());
}


int main(int argc, char *argv[]) {

    ros::init(argc, argv, "point_cloud_scene_decomposer");
    srand(time(NULL));
    PointCloudSceneDecomposer pcsd;
    ros::spin();
    return 0;
}
