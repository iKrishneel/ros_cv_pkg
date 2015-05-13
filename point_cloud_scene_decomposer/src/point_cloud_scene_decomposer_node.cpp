
#include <point_cloud_scene_decomposer/point_cloud_scene_decomposer.h>

PointCloudSceneDecomposer::PointCloudSceneDecomposer() :
    normal_(pcl::PointCloud<pcl::Normal>::Ptr(
               new pcl::PointCloud<pcl::Normal>)) {
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
    this->sub_image_ = nh_.subscribe("input_image", 1,
        &PointCloudSceneDecomposer::imageCallback, this);
    // this->sub_norm_ = nh_.subscribe("input_norm", 1,
    //    &PointCloudSceneDecomposer::normalCallback, this);
    this->sub_cloud_ = nh_.subscribe("input_cloud", 1,
        &PointCloudSceneDecomposer::cloudCallback, this);
}

void PointCloudSceneDecomposer::unsubscribe() {
    this->sub_cloud_.shutdown();
    this->sub_norm_.shutdown();
    this->sub_image_.shutdown();
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

    std::cout << cloud->size() << "\t" << normal_->size() << image_.size() << std::endl;
    if (cloud->empty() || this->image_.empty()) {
       ROS_ERROR("CANNOT PROCESS EMPTY INSTANCE");
       return;
    }
    // cv::Mat rgb_img;
    // cv::Mat dep_img;
    // this->pointCloud2RGBDImage(cloud, rgb_img, dep_img);

    cv::Mat image = this->image_.clone();
    cv::Mat depth_edge;
    this->getRGBEdge(image, depth_edge, "cvSOBEL");
    // this->getDepthEdge(dep_img, depth_edge, true);
    cv::imshow("depth_edge", depth_edge);
    
    
    std::vector<cvPatch<int> > patch_label;
    this->cvLabelEgdeMap(cloud, depth_edge, patch_label);
    
    cv::waitKey(3);
    
    /*
    pcl::PointCloud<PointT>::Ptr patch_cloud(
       new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT> (
       *cloud, *patch_cloud);
    std::vector<pcl::PointCloud<PointT>::Ptr> cloud_clusters;
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normal_clusters;
    pcl::PointCloud<pcl::PointXYZ>::Ptr centroids(
       new pcl::PointCloud<pcl::PointXYZ>);
    this->extractPointCloudClustersFrom2DMap(
       patch_cloud, patch_label, cloud_clusters, normal_clusters, centroids);
    std::vector<std::vector<int> > neigbour_idx;
    this->pclNearestNeigborSearch(centroids, neigbour_idx, false, 8, 0.05);

    RegionAdjacencyGraph *rag = new RegionAdjacencyGraph();
    rag->generateRAG(
       cloud_clusters, normal_clusters, centroids, neigbour_idx, 1);
    rag->splitMergeRAG(0.0);
    std::vector<int> labelMD;
    rag->getCloudClusterLabels(labelMD);
    // std::map<int, pcl::PointCloud<PointT>::Ptr> c_clusters;
    // rag->concatenateRegionUsingRAGInfo(
    //    cloud_clusters, normal_clusters, c_clusters);
    free(rag);
    this->semanticCloudLabel(cloud_clusters, cloud, labelMD);

    /*
    // labeling by convex criteria
    normal_clusters.clear();
    cloud_clusters.clear();
    centroids->clear();
    neigbour_idx.clear();
    for (std::map<int, pcl::PointCloud<PointT>::Ptr>::iterator it =
            c_clusters.begin(); it != c_clusters.end(); it++) {
       pcl::PointCloud<PointT>::Ptr m_cloud((*it).second);
       m_cloud->header = cloud->header;
       pcl::PointCloud<pcl::Normal>::Ptr s_normal(
          new pcl::PointCloud<pcl::Normal>);
       this->estimatePointCloudNormals(m_cloud, s_normal, 40, 0.05, false);
       
       Eigen::Vector4f centroid;
       pcl::compute3DCentroid<PointT, float>(*m_cloud, centroid);
       float ct_x = static_cast<float>(centroid[0]);
       float ct_y = static_cast<float>(centroid[1]);
       float ct_z = static_cast<float>(centroid[2]);
       
       centroids->push_back(pcl::PointXYZ(ct_x, ct_y, ct_z));
       cloud_clusters.push_back(m_cloud);
       normal_clusters.push_back(s_normal);
    }
    this->pclNearestNeigborSearch(centroids, neigbour_idx, true, 4);
 
    rag = new RegionAdjacencyGraph();
    rag->generateRAG(
       cloud_clusters, normal_clusters, centroids, neigbour_idx, 1);
    rag->splitMergeRAG(0.0f);
    // std::vector<int> labelMD;
    labelMD.clear();
    rag->getCloudClusterLabels(labelMD);
    free(rag);
    this->semanticCloudLabel(cloud_clusters, cloud, labelMD);
    
    /**/
    // this->viewPointSurfaceNormalOrientation(cloud, this->normal_);
    // this->pointCloudLocalGradient(cloud, this->normal_, dep_img);
    //
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}


// void PointCloudSceneDecomposer::runSceneDecomposer() {
    
// }

/**
 * Function to compute the point cloud normal
 */
void PointCloudSceneDecomposer::estimatePointCloudNormals(
    const pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<pcl::Normal>::Ptr s_normal,
    const int k,
    const double radius,
    bool ksearch) {
    if (cloud->empty()) {
       ROS_ERROR("ERROR: The Input cloud is Empty.....");
       return;
    }
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setNumberOfThreads(8);
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
    ne.setSearchMethod(tree);
    if (ksearch) {
        ne.setKSearch(k);
    } else {
        ne.setRadiusSearch(radius);
    }
    ne.compute(*s_normal);
}

/**
 * function to estimate the nearest neigbour normals
 */
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
 * extract 2D rgb and depth image from the point cloud data
 */
void PointCloudSceneDecomposer::pointCloud2RGBDImage(
     pcl::PointCloud<PointT>::Ptr _cloud,
     cv::Mat &rgbImage,
     cv::Mat &depthImage) {
     depthImage = cv::Mat(_cloud->height, _cloud->width, CV_8U);
     rgbImage = cv::Mat::zeros(_cloud->height, _cloud->width, CV_8UC3);
     for (int j = 0; j < _cloud->height; j++) {
        for (int i = 0; i < _cloud->width; i++) {
           int index = i + (j * _cloud->width);
           float distance_ = _cloud->points[index].z;
           if (distance_ != distance_) {
              depthImage.at<uchar>(j, i) = 0.0f;

              rgbImage.at<cv::Vec3b>(j, i)[2] = 0.0f;
              rgbImage.at<cv::Vec3b>(j, i)[1] = 0.0f;
              rgbImage.at<cv::Vec3b>(j, i)[0] = 0.0f;
           } else {
              depthImage.at<uchar>(j, i) = (
                 distance_ / 10.0f) * 255;
              
              rgbImage.at<cv::Vec3b>(j, i)[2] = _cloud->points[index].r;
              rgbImage.at<cv::Vec3b>(j, i)[1] = _cloud->points[index].g;
              rgbImage.at<cv::Vec3b>(j, i)[0] = _cloud->points[index].b;
           }
        }
     }
     // cv::imshow("rgb", rgbImage);
     cv::imshow("depth", depthImage);
     cv::waitKey(3);
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

/**
 * compute the normal orientation with respect to the viewpoint
 */
void PointCloudSceneDecomposer::viewPointSurfaceNormalOrientation(
    pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr cloud_normal,
    cv::Mat &n_edge) {
    if (cloud->empty() || cloud_normal->empty()) {
        ROS_ERROR("ERROR: Point Cloud | Normal vector is empty...");
        return;
    }
    pcl::PointCloud<PointT>::Ptr gradient_cloud(new pcl::PointCloud<PointT>);
    for (int i = 0; i < cloud->size(); i++) {
       Eigen::Vector3f viewPointVec =
          cloud->points[i].getVector3fMap();
       Eigen::Vector3f surfaceNormalVec = Eigen::Vector3f(
          -cloud_normal->points[i].normal_x,
          -cloud_normal->points[i].normal_y,
          -cloud_normal->points[i].normal_z);
       float cross_norm = static_cast<float>(
          surfaceNormalVec.cross(viewPointVec).norm());
       float scalar_prod = static_cast<float>(
          surfaceNormalVec.dot(viewPointVec));
       float angle = atan2(cross_norm, scalar_prod);
       if (angle * (180/CV_PI) >= 0 && angle * (180/CV_PI) <= 180) {
          cv::Scalar jmap = JetColour(angle/(2*CV_PI), 0, 1);
          PointT pt;
          pt.x = cloud->points[i].x;
          pt.y = cloud->points[i].y;
          pt.z = cloud->points[i].z;
          pt.r = jmap.val[0] * 255;
          pt.g = jmap.val[1] * 255;
          pt.b = jmap.val[2] * 255;
          gradient_cloud->push_back(pt);
       }
    }
    cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*gradient_cloud, *cloud);

/*    
    cv::Mat normalMap = cv::Mat::zeros(
       cv::Size(cloud->width, cloud->height), CV_8UC3);
    for (int j = 0; j < cloud->height; j++) {
       for (int i = 0; i < cloud->width; i++) {
          int vp_index = i + (j * cloud->width);
          Eigen::Vector3f viewPointVec =
             cloud->points[vp_index].getVector3fMap();
          Eigen::Vector3f surfaceNormalVec = Eigen::Vector3f(
             -cloud_normal->points[vp_index].normal_x,
             -cloud_normal->points[vp_index].normal_y,
             -cloud_normal->points[vp_index].normal_z);
          float cross_norm = static_cast<float>(
             surfaceNormalVec.cross(viewPointVec).norm());
          float scalar_prod = static_cast<float>(
             surfaceNormalVec.dot(viewPointVec));
          float angle = atan2(cross_norm, scalar_prod);
          
          if (angle * (180/CV_PI) >= 0 && angle * (180/CV_PI) <= 180) {
             cv::Scalar jmap = JetColour(angle/(2*CV_PI), 0, 1);
             normalMap.at<cv::Vec3b>(j, i)[0] = jmap.val[0] * 255;
             normalMap.at<cv::Vec3b>(j, i)[1] = jmap.val[1] * 255;
             normalMap.at<cv::Vec3b>(j, i)[2] = jmap.val[2] * 255;
          }
       }
    }
    this->getRGBEdge(normalMap, n_edge, "cvSOBEL");
    // cv::imshow("ViewPoint Normal Edge", n_edge);
    cv::imshow("ViewPoint Normal Map", normalMap);
*/
}

/**
 * project the 2D region map to 3D point cloud and uniquely label each patch
 */
void PointCloudSceneDecomposer::extractPointCloudClustersFrom2DMap(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const std::vector<cvPatch<int> > &patch_label,
    std::vector<pcl::PointCloud<PointT>::Ptr> &cloud_clusters,
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> &normal_clusters,
    pcl::PointCloud<pcl::PointXYZ>::Ptr centroids) {
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
       for (int j = 0; j < rect.height; j++) {
          for (int i = 0; i < rect.width; i++) {
             int label_ = static_cast<int>(
                labelMD.at<float>(j, i));
             int index = (i + rect.x) + ((j + rect.y) * cloud->width);
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
             pcl::compute3DCentroid<PointT, float>(*cloud, *indices, centroid);
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
       cluster_indices.clear();
    }
    // std::cout << "--INFO: Cluster Size: "
    //           << cloud_clusters.size() << std::endl;
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



int main(int argc, char *argv[]) {

    ros::init(argc, argv, "my_pcl_tutorial");
    srand(time(NULL));
    PointCloudSceneDecomposer pcsd;
    ros::spin();
    return 0;
}

