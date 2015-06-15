
#include <point_cloud_scene_decomposer/point_cloud_scene_decomposer.h>
#include <point_cloud_scene_decomposer/ClusterVoxels.h>
#include <vector>

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

    /**
     * Do Clustering here
       <x/z, y/z, 1/z, nx, ny, nz, norm_vp, r/255, g/255, b/255>
     */
    std::vector<int> labelMD;
    this->pointCloudVoxelClustering(
       cloud_clusters, normal_clusters, centroids, labelMD);
    this->semanticCloudLabel(cloud_clusters, cloud, labelMD);
    
    
    /*
    std::vector<std::vector<int> > neigbour_idx;
    this->pclNearestNeigborSearch(centroids, neigbour_idx, true, 3, 0.06);
    
    RegionAdjacencyGraph *rag = new RegionAdjacencyGraph();
    rag->generateRAG(
       cloud_clusters, normal_clusters, centroids, neigbour_idx, 1);
    rag->splitMergeRAG(0.10);
    std::vector<int> labelMD;
    rag->getCloudClusterLabels(labelMD);
    free(rag);
    this->semanticCloudLabel(cloud_clusters, cloud, labelMD);
    
    // this->pointCloudLocalGradient(cloud, this->normal_, dep_img);
    */
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

/**
 * label the region
 */
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


/**
 * 
 */
void PointCloudSceneDecomposer::pointCloudVoxelClustering(
    std::vector<pcl::PointCloud<PointT>::Ptr> & cloud_clusters,
    const std::vector<pcl::PointCloud<pcl::Normal>::Ptr> &normal_clusters,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centroids,
    std::vector<int> &labelMD) {
    if (cloud_clusters.size() != normal_clusters.size() &&
        cloud_clusters.size() != cloud_centroids->size()) {
       ROS_ERROR("--CANNOT CLUSTER VOXEL");
       return;
    }
    int icounter = 0;
    const int dimensionality = 4;
    pcl::PointCloud<pcl::PointXYZ>::Ptr centroids(
       new pcl::PointCloud<pcl::PointXYZ>);
    cv::Mat cluster_features = cv::Mat(
       static_cast<int>(cloud_clusters.size()), dimensionality, CV_32F);
    for (std::vector<pcl::PointCloud<PointT>::Ptr>::iterator it =
            cloud_clusters.begin(); it != cloud_clusters.end(); it++) {
       Eigen::Vector3f cluster_centroid =
          cloud_centroids->points[icounter].getVector3fMap();
       // assign centroid to cloud value
       int index = -1;
       float distance = FLT_MAX;
       for (int i = 0; i < (*it)->size(); i++) {
          Eigen::Vector3f cloud_pt = (*it)->points[i].getVector3fMap();
          float x_val = pow(cluster_centroid(0) - cloud_pt(0), 2);
          float y_val = pow(cluster_centroid(1) - cloud_pt(1), 2);
          float z_val = pow(cluster_centroid(2) - cloud_pt(2), 2);
          float dist = sqrt(x_val + y_val + z_val);
          if (dist < distance) {
             distance = dist;
             index = i;
          }
       }
       Eigen::Vector3f cloud_pt = (*it)->points[index].getVector3fMap();
       centroids->push_back(pcl::PointXYZ(
                               cloud_pt(0), cloud_pt(1), cloud_pt(2)));

       
       
       Eigen::Vector3f normal_pt = Eigen::Vector3f(
          normal_clusters[icounter]->points[index].normal_x,
          normal_clusters[icounter]->points[index].normal_y,
          normal_clusters[icounter]->points[index].normal_z);

       normal_pt(0) = isnan(normal_pt(0)) ? 0 : normal_pt(0);
       normal_pt(1) = isnan(normal_pt(1)) ? 0 : normal_pt(1);
       normal_pt(2) = isnan(normal_pt(2)) ? 0 : normal_pt(2);
       
       float cross_norm = static_cast<float>(
          normal_pt.cross(cloud_pt).norm());
       float scalar_prod = static_cast<float>(
          normal_pt.dot(cloud_pt));
       float angle = atan2(cross_norm, scalar_prod);

       int f_ind = 0;
       cluster_features.at<float>(icounter, f_ind++) = cloud_pt(0)/cloud_pt(2);
       cluster_features.at<float>(icounter, f_ind++) = cloud_pt(1)/cloud_pt(2);
       // cluster_features.at<float>(icounter, ++f_ind) = cloud_pt(2);
       // cluster_features.at<float>(icounter, ++f_ind) = normal_pt(0);
       // cluster_features.at<float>(icounter, ++f_ind) = normal_pt(1);
       // cluster_features.at<float>(icounter, ++f_ind) = normal_pt(2);
       // cluster_features.at<float>(icounter, ++f_ind) = angle;
       cluster_features.at<float>(icounter, f_ind++) =
          (*it)->points[index].r/255.0f;
       cluster_features.at<float>(icounter, f_ind++) =
          (*it)->points[index].g/255.0f;
       cluster_features.at<float>(icounter, f_ind++) =
          (*it)->points[index].b/255.0f;
       icounter++;
    }

    
    std::vector<std::vector<int> > neigbour_index;
    this->pclNearestNeigborSearch(centroids, neigbour_index, true, 3, 0.05);
    for (int i = 0; i < neigbour_idx[0].size(); i++) {
       
    }
    
    
    int cluster_size = 20;
    cv::Mat labels;
    int attempts = 1000;
    cv::Mat centers;
    cv::TermCriteria criteria = cv::TermCriteria(
       CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100000, 0.00001);

    std::cout << cluster_features.size() << std::endl;
    // std::cout << std::endl;

    this->clusterVoxels(cluster_features, labelMD);
    /*
    if (cluster_size < cluster_features.rows) {
       cv::kmeans(cluster_features, cluster_size, labels, criteria,
                  attempts, cv::KMEANS_RANDOM_CENTERS, centers);
    }
    for (int i = 0; i < labels.rows; i++) {
       labelMD.push_back(labels.at<int>(i, 0));
    }
    */
}

void PointCloudSceneDecomposer::clusterVoxels(
    const cv::Mat &voxel_features, std::vector<int> &labelMD) {
    if (voxel_features.empty()) {
       ROS_ERROR("--EMPTY FEATURE VECTOR");
       return;
    }
    point_cloud_scene_decomposer::ClusterVoxels srv_cv;
    for (int j = 0; j < voxel_features.rows; j++) {
       for (int i = 0; i < voxel_features.cols; i++) {
          float element = voxel_features.at<float>(j, i);
          srv_cv.request.features.push_back(element);
       }
    }
    srv_cv.request.stride = static_cast<int>(voxel_features.cols);
    if (this->srv_client_.call(srv_cv)) {
       for (int i = 0; i < voxel_features.rows; i++) {
          int label = srv_cv.response.labels[i];
          labelMD.push_back(static_cast<int>(label));
       }
    } else {
       ROS_ERROR("Failed to call module for clustering");
       return;
    }
}

int main(int argc, char *argv[]) {

    ros::init(argc, argv, "point_cloud_scene_decomposer");
    srand(time(NULL));
    PointCloudSceneDecomposer pcsd;
    ros::spin();
    return 0;
}

