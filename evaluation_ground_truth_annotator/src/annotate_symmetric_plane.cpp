
#include <evaluation_ground_truth_annotator/annotate_symmetric_plane.h>

AnnotateSymmetricPlane::AnnotateSymmetricPlane() {
    this->selected_points2d_.clear();
    is_init_ = true;

    this->onInit();
    
}

void AnnotateSymmetricPlane::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/plane_normal", 1);
}

void AnnotateSymmetricPlane::subscribe() {
    this->screen_pt_ = this->pnh_.subscribe(
       "input_point", 1, &AnnotateSymmetricPlane::screenCB, this);
    this->sub_cloud_ = this->pnh_.subscribe(
       "input_cloud", 1, &AnnotateSymmetricPlane::cloudCB, this);
}

void AnnotateSymmetricPlane::screenCB(
    const geometry_msgs::PointStamped::ConstPtr &screen_msg) {
    int x = screen_msg->point.x;
    int y = screen_msg->point.y;
    this->selected_points2d_.push_back(cv::Point2i(x, y));
}


void AnnotateSymmetricPlane::cloudCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {
    if (this->selected_points2d_.size() < 3) {
       ROS_ERROR("NUMBER OF POINTS SELECTED IS < 3: %d",
                 selected_points2d_.size());
       return;
    }

    ROS_WARN("COMPUTING PLANE NORM");
    
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    Eigen::Vector3f point3d[3];
    for (int i = 0; i < selected_points2d_.size(); i++) {
       int index = selected_points2d_[i].x + (cloud->width *
                                              selected_points2d_[i].y);
       point3d[i] = cloud->points[index].getVector3fMap();
    }

    Eigen::Vector3f pta = point3d[2] - point3d[0];
    Eigen::Vector3f ptb = point3d[1] - point3d[0];
    Eigen::Vector3f cross = pta.cross(ptb);

    pcl::PointXYZRGBNormal pt;
    pt.x = point3d[0](0);
    pt.y = point3d[0](1);
    pt.z = point3d[0](2);
    pt.r = 255;
    pt.normal_x = cross(0);
    pt.normal_y = cross(1);
    pt.normal_z = cross(2);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr norm_points(
       new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    norm_points->push_back(pt);
    norm_points->push_back(pt);
    norm_points->push_back(pt);

    
    this->selected_points2d_.clear();
    is_init_ = !is_init_;

    sensor_msgs::PointCloud2 *ros_cloud = new sensor_msgs::PointCloud2;
    pcl::toROSMsg(*norm_points, *ros_cloud);
    ros_cloud->header = cloud_msg->header;
    this->pub_cloud_.publish(*ros_cloud);
    
}



int main(int argc, char *argv[]) {
    ros::init(argc, argv, "annotate_symmetric_plane");
    AnnotateSymmetricPlane asp;
    ros::spin();
    return 0;
}

