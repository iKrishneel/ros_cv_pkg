
#include <point_cloud_normal_gradient/point_cloud_normal_gradient.h>
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
    
    pcl::PointCloud<pcl::Normal>::Ptr normals(
       new pcl::PointCloud<pcl::Normal>);
    this->estimatePointCloudNormals(cloud, normals, 30, 0.05, false);
    this->viewPointSurfaceNormalOrientation(cloud, normals);

    // add module if want to see normal on rviz
    
    sensor_msgs::PointCloud2 ros_normal;
    pcl::toROSMsg(*normals, ros_normal);
    ros_normal.header = cloud_msg->header;
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;

    this->pub_norm_.publish(ros_normal);
    this->pub_cloud_.publish(ros_cloud);
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
