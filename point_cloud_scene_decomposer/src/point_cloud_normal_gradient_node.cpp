
#include <point_cloud_scene_decomposer/point_cloud_normal_gradient.h>
#include <iostream>

PointCloudNormalGradients::PointCloudNormalGradients() {
    this->subscribe();
    this->onInit();
}

void PointCloudNormalGradients::onInit() {
    this->pub_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>(
        "/normal_gradient/output/cloud", sizeof(char));
    this->pub_norm_ = nh_.advertise<sensor_msgs::PointCloud2>(
        "/normal_gradient/output/normal", sizeof(char));
    this->pub_norm_xyz_ = nh_.advertise<sensor_msgs::PointCloud2>(
        "/normal_gradient/output/normal_xyz_", sizeof(char));
}

void PointCloudNormalGradients::subscribe() {
    this->sub_cloud_ = nh_.subscribe("input", 1,
       &PointCloudNormalGradients::cloudCallback, this);
}

void PointCloudNormalGradients::unsubscribe() {
    this->sub_cloud_.shutdown();
}

void PointCloudNormalGradients::cloudCallback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    std::vector< int > index;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, index);
    
    pcl::PointCloud<pcl::Normal>::Ptr normals(
       new pcl::PointCloud<pcl::Normal>);
    this->estimatePointCloudNormals(cloud, normals, 30, 0.05, false);

    pcl::PointCloud<PointT>::Ptr norm_grad_cloud(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(*cloud, *norm_grad_cloud);
    this->viewPointSurfaceNormalOrientation(norm_grad_cloud, normals);

    // pcl::PointCloud<PointT>::Ptr curvature_cloud(new pcl::PointCloud<PointT>);
    // pcl::copyPointCloud<PointT, PointT>(*cloud, *curvature_cloud);
    // this->localCurvatureBoundary(curvature_cloud, normals);
    
    // module to see normal on rviz
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr normal_xyz(
       new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    this->convertToRvizNormalDisplay(cloud, normals, normal_xyz);
    
    sensor_msgs::PointCloud2 ros_normal;
    pcl::toROSMsg(*normals, ros_normal);
    ros_normal.header = cloud_msg->header;
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*norm_grad_cloud, ros_cloud);
    // pcl::toROSMsg(*curvature_cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;

    sensor_msgs::PointCloud2 ros_normal_xyz_cloud;
    pcl::toROSMsg(*normal_xyz, ros_normal_xyz_cloud);
    ros_normal_xyz_cloud.header = cloud_msg->header;
    
    this->pub_norm_.publish(ros_normal);
    this->pub_cloud_.publish(ros_cloud);
    this->pub_norm_xyz_.publish(ros_normal_xyz_cloud);
}


void PointCloudNormalGradients::estimatePointCloudNormals(
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

void PointCloudNormalGradients::viewPointSurfaceNormalOrientation(
    pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr cloud_normal) {
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
}

void PointCloudNormalGradients::localCurvatureBoundary(
    pcl::PointCloud<PointT>::Ptr cloud,
    pcl::PointCloud<pcl::Normal>::Ptr normal) {
    if (cloud->empty() || normal->empty() ||
        (cloud->size() != normal->size())) {
        ROS_ERROR("ERROR: Point Cloud | Normal vector is empty...");
        return;
    }
    std::vector<std::vector<int> > neigbour_idx;
    this->pclNearestNeigborSearch(cloud, neigbour_idx, true, 40, 0.05);
    pcl::PointCloud<PointT>::Ptr curvature_cloud(new pcl::PointCloud<PointT>);
    for (int i = 0; i < cloud->size(); i++) {
       Eigen::Vector3f centerPointVec = Eigen::Vector3f(
          normal->points[i].normal_x,
          normal->points[i].normal_y,
          normal->points[i].normal_z);
       int icounter = 0;
       float scalarProduct = 0.0f;
       float max_difference__ = 0.0;
       float min_difference__ = sizeof(short) * M_PI;
       for (std::vector<int>::iterator it = neigbour_idx[i].begin();
            it != neigbour_idx[i].end(); it++) {
          Eigen::Vector3f neigbourPointVec = Eigen::Vector3f(
             normal->points[*it].normal_x,
             normal->points[*it].normal_y,
             normal->points[*it].normal_z);
          float scalarProduct = (
             neigbourPointVec.dot(centerPointVec) /
             (neigbourPointVec.norm() * centerPointVec.norm()));

          if (scalarProduct > max_difference__) {
             max_difference__ = scalarProduct;
          }
          if (scalarProduct < min_difference__) {
            min_difference__ = scalarProduct;
         }
       }
       // scalarProduct /= static_cast<float>(icounter);
       float variance = /*max_difference__*/  -min_difference__;
       if (variance < 0.10f) {
          // variance = 0.0;
       }
       // cv::Scalar jmap = JetColour<float, float, float>(variance, 0, 1);
       PointT pt;
       pt.x = cloud->points[i].x;
       pt.y = cloud->points[i].y;
       pt.z = cloud->points[i].z;
       pt.r = 0;  // jmap.val[0] * 255;
       pt.g = 0;  // jmap.val[1] * 255;
       pt.b = variance * 255;  //  jmap.val[2] * 255;
       curvature_cloud->push_back(pt);
    }
    cloud->clear();
    pcl::copyPointCloud<PointT, PointT>(*curvature_cloud, *cloud);
}

void PointCloudNormalGradients::pclNearestNeigborSearch(
    pcl::PointCloud<PointT>::Ptr cloud,
     std::vector<std::vector<int> > &pointIndices,
     bool isneigbour, const int k, const double radius) {
    if (cloud->empty()) {
       ROS_ERROR("Cannot search NN in an empty point cloud");
       return;
    }
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);
    std::vector<std::vector<float> > pointSquaredDistance;
    for (int i = 0; i < cloud->size(); i++) {
       std::vector<int>pointIdx;
       std::vector<float> pointSqDist;
       PointT searchPoint = cloud->points[i];
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

void PointCloudNormalGradients::convertToRvizNormalDisplay(
    const pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr normals,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr normal_xyz) {
       normal_xyz->points.resize(cloud->points.size());
    for (size_t i = 0; i < normal_xyz->points.size(); i++) {
       pcl::PointXYZRGBNormal p;
       p.x = cloud->points[i].x;
       p.y = cloud->points[i].y;
       p.z = cloud->points[i].z;
       p.normal_x = normals->points[i].normal_x;
       p.normal_y = normals->points[i].normal_y;
       p.normal_z = normals->points[i].normal_z;
       normal_xyz->points[i] = p;
    }
}


template<typename T, typename U, typename V>
cv::Scalar PointCloudNormalGradients::JetColour(T v, U vmin, V vmax) {
    cv::Scalar c = cv::Scalar(1.0, 1.0, 1.0);  // white
    T dv;
    if (v < vmin)
       v = vmin;
    if (v > vmax)
       v = vmax;
    dv = vmax - vmin;
    if (v < (vmin + 0.25 * dv)) {
       c.val[0] = 0;
       c.val[1] = 4 * (v - vmin) / dv;
    } else if (v < (vmin + 0.5 * dv)) {
       c.val[0] = 0;
       c.val[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    } else if (v < (vmin + 0.75 * dv)) {
       c.val[0] = 4 * (v - vmin - 0.5 * dv) / dv;
       c.val[2] = 0;
    } else {
       c.val[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
       c.val[2] = 0;
    }
    return(c);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "point_cloud_normal_gradient");
    PointCloudNormalGradients pcng;
    ros::spin();
    return 0;
}
