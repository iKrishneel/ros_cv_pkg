
#include <cuboid_bilateral_symmetric_segmentation/oriented_bounding_box.h>

OrientedBoundingBox::OrientedBoundingBox() :
    use_pca_(true), force_to_flip_z_axis_(true), align_boxes_(true) {
   
}

bool OrientedBoundingBox::fitOriented3DBoundingBox(
    jsk_msgs::BoundingBox &bounding_box,
    const pcl::PointCloud<PointT>::Ptr cloud,
    const jsk_msgs::PolygonArrayConstPtr &planes,
    const jsk_msgs::ModelCoefficientsArrayConstPtr &coefficients) {
    if (cloud->empty()) {
       ROS_ERROR("EMPTY DATA FOR FITTING BOUNDING BOX");
       return false;
    }
    Eigen::Vector4f center;
    pcl::compute3DCentroid(*cloud, center);
    bool successp = computeBoundingBox(cloud, center, planes,
                                       coefficients, bounding_box);
    if (!successp) {
       return false;
    }
    return true;
}

bool OrientedBoundingBox::computeBoundingBox(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmented_cloud,
    const Eigen::Vector4f center, const jsk_msgs::PolygonArrayConstPtr& planes,
    const jsk_msgs::ModelCoefficientsArrayConstPtr& coefficients,
    jsk_msgs::BoundingBox& bounding_box) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
      segmented_cloud_transformed (new pcl::PointCloud<pcl::PointXYZRGB>);
    // align boxes if possible
    Eigen::Matrix4f m4 = Eigen::Matrix4f::Identity();
    Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
    if (align_boxes_) {
      int nearest_plane_index = findNearestPlane(center, planes, coefficients);
      if (nearest_plane_index == -1) {
        segmented_cloud_transformed = segmented_cloud;
        ROS_ERROR("no planes to align boxes are given");
      } else {
        Eigen::Vector3f normal, z_axis;
        if (force_to_flip_z_axis_) {
           normal[0] = - coefficients->coefficients[
              nearest_plane_index].values[0];
           normal[1] = - coefficients->coefficients[
              nearest_plane_index].values[1];
           normal[2] = - coefficients->coefficients[
              nearest_plane_index].values[2];
        } else {
           normal[0] = coefficients->coefficients[
              nearest_plane_index].values[0];
           normal[1] = coefficients->coefficients[
              nearest_plane_index].values[1];
           normal[2] = coefficients->coefficients[
              nearest_plane_index].values[2];
        }
        normal = normal.normalized();
        Eigen::Quaternionf rot;
        rot.setFromTwoVectors(Eigen::Vector3f::UnitZ(), normal);
        Eigen::AngleAxisf rotation_angle_axis(rot);
        Eigen::Vector3f rotation_axis = rotation_angle_axis.axis();
        double theta = rotation_angle_axis.angle();
        if (isnan(theta) || isnan(rotation_axis[0]) ||
            isnan(rotation_axis[1]) || isnan(rotation_axis[2])) {
           segmented_cloud_transformed = segmented_cloud;
           ROS_ERROR("cannot compute angle to align the point cloud:");
           ROS_ERROR("[%f, %f, %f], [%f, %f, %f]", z_axis[0], z_axis[1],
                     z_axis[2], normal[0], normal[1], normal[2]);
        } else {
          Eigen::Matrix3f m = Eigen::Matrix3f::Identity() * rot;
          if (use_pca_) {
            // first project points to the plane
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr projected_cloud
              (new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::ProjectInliers<pcl::PointXYZRGB> proj;
            proj.setModelType(pcl::SACMODEL_PLANE);
            pcl::ModelCoefficients::Ptr
              plane_coefficients (new pcl::ModelCoefficients);
            plane_coefficients->values
              = coefficients->coefficients[nearest_plane_index].values;
            proj.setModelCoefficients(plane_coefficients);
            proj.setInputCloud(segmented_cloud);
            proj.filter(*projected_cloud);
            if (projected_cloud->points.size() >= 3) {
              pcl::PCA<pcl::PointXYZRGB> pca;
              pca.setInputCloud(projected_cloud);
              Eigen::Matrix3f eigen = pca.getEigenVectors();
              m.col(0) = eigen.col(0);
              m.col(1) = eigen.col(1);
              // flip axis to satisfy right-handed system
              if (m.col(0).cross(m.col(1)).dot(m.col(2)) < 0) {
                m.col(0) = - m.col(0);
              }
              if (m.col(0).dot(Eigen::Vector3f::UnitX()) < 0) {
                // rotate around z
                m = m * Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ());
              }
            } else {
              ROS_ERROR("Too small indices for PCA computation");
              return false;
            }
          }
            
          // m4 <- m
          for (size_t row = 0; row < 3; row++) {
            for (size_t column = 0; column < 3; column++) {
              m4(row, column) = m(row, column);
            }
          }
          q = m;
          Eigen::Matrix4f inv_m = m4.inverse();
          pcl::transformPointCloud(*segmented_cloud,
                                   *segmented_cloud_transformed, inv_m);
        }
      }
    } else {
      segmented_cloud_transformed = segmented_cloud;
    }
      
    // create a bounding box
    Eigen::Vector4f minpt, maxpt;
    pcl::getMinMax3D<pcl::PointXYZRGB>(*segmented_cloud_transformed,
                                       minpt, maxpt);

    double xwidth = maxpt[0] - minpt[0];
    double ywidth = maxpt[1] - minpt[1];
    double zwidth = maxpt[2] - minpt[2];
    
    Eigen::Vector4f center2((maxpt[0] + minpt[0]) / 2.0,
                            (maxpt[1] + minpt[1]) / 2.0,
                            (maxpt[2] + minpt[2]) / 2.0, 1.0);
    Eigen::Vector4f center_transformed = m4 * center2;
    
    bounding_box.pose.position.x = center_transformed[0];
    bounding_box.pose.position.y = center_transformed[1];
    bounding_box.pose.position.z = center_transformed[2];
    bounding_box.pose.orientation.x = q.x();
    bounding_box.pose.orientation.y = q.y();
    bounding_box.pose.orientation.z = q.z();
    bounding_box.pose.orientation.w = q.w();
    bounding_box.dimensions.x = xwidth;
    bounding_box.dimensions.y = ywidth;
    bounding_box.dimensions.z = zwidth;
    return true;
  }

int OrientedBoundingBox::findNearestPlane(
    const Eigen::Vector4f& center,
    const jsk_msgs::PolygonArrayConstPtr& planes,
    const jsk_msgs::ModelCoefficientsArrayConstPtr& coefficients) {
    double min_distance = DBL_MAX;
    int nearest_index = -1;
    for (size_t i = 0; i < coefficients->coefficients.size(); i++) {
      geometry_msgs::PolygonStamped polygon_msg = planes->polygons[i];
      jsk_recognition_utils::Vertices vertices;
      for (size_t j = 0; j < polygon_msg.polygon.points.size(); j++) {
         jsk_recognition_utils::Vertex v;
         v[0] = polygon_msg.polygon.points[j].x;
         v[1] = polygon_msg.polygon.points[j].y;
         v[2] = polygon_msg.polygon.points[j].z;
         vertices.push_back(v);
      }
      jsk_recognition_utils::ConvexPolygon p(
         vertices, coefficients->coefficients[i].values);
      double distance = p.distanceToPoint(center);
      if (distance < min_distance) {
         min_distance = distance;
         nearest_index = i;
      }
    }
    return nearest_index;
}

void OrientedBoundingBox::transformBoxCornerPoints(
    pcl::PointCloud<PointT>::Ptr cloud,
    const jsk_msgs::BoundingBox bounding_box) {
    Eigen::Vector3f center = Eigen::Vector3f(bounding_box.pose.position.x,
                                             bounding_box.pose.position.y,
                                             bounding_box.pose.position.z);
    Eigen::Vector3f dims = Eigen::Vector3f(bounding_box.dimensions.x / 2.0f,
                                           bounding_box.dimensions.y / 2.0f,
                                           bounding_box.dimensions.z / 2.0f);
    const int SIZE = 12;
    Eigen::Vector3f corners[SIZE];
    // 4 corners
    corners[0] = Eigen::Vector3f(dims(0), dims(1), dims(2));
    corners[1] = Eigen::Vector3f(-dims(0), -dims(1), -dims(2));
    corners[2] = Eigen::Vector3f(-dims(0), -dims(1), dims(2));
        
    corners[3] = Eigen::Vector3f(dims(0), -dims(1), dims(2));
    corners[4] = Eigen::Vector3f(-dims(0), dims(1), -dims(2));
    corners[5] = Eigen::Vector3f(-dims(0), dims(1), dims(2));
    
    // 4 mid-points
    corners[6] = Eigen::Vector3f(dims(0), 0, -dims(2));
    corners[7] = Eigen::Vector3f(-dims(0), 0, -dims(2));
    corners[8] = Eigen::Vector3f(-dims(0), 0, dims(2));
    
    corners[9] = Eigen::Vector3f(0, -dims(1), -dims(2));
    corners[10] = Eigen::Vector3f(0, dims(1), dims(2));
    corners[11] = Eigen::Vector3f(0, dims(1), -dims(2));
    
    Eigen::Vector3f colors[SIZE];
    colors[0] = Eigen::Vector3f(255, 0, 0);  // D2
    colors[1] = Eigen::Vector3f(255, 0, 0);  // D2
    colors[2] = Eigen::Vector3f(255, 0, 0);  // D2
    
    colors[3] = Eigen::Vector3f(0, 255, 0);  // D1
    colors[4] = Eigen::Vector3f(0, 255, 0);  // D1
    colors[5] = Eigen::Vector3f(0, 255, 0);  // D1
    
    colors[6] = Eigen::Vector3f(0, 0, 255);  // C1
    colors[7] = Eigen::Vector3f(0, 0, 255);  // C1
    colors[8] = Eigen::Vector3f(0, 0, 255);  // C1

    colors[9] = Eigen::Vector3f(255, 0, 255);  // C1
    colors[10] = Eigen::Vector3f(255, 0, 255);  // C2
    colors[11] = Eigen::Vector3f(255, 0, 255);  // C2
    
    cloud->clear();
    for (int i = 0; i < SIZE; i++) {
       PointT pt = this->Eigen2PointT(corners[i], colors[i]);
       cloud->push_back(pt);
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
    
    pcl::transformPointCloud(*cloud, *cloud, transform);
    PointT pt = this->Eigen2PointT(center, Eigen::Vector3f(255, 255, 255));
    cloud->push_back(pt);

    this->plotPlane(cloud);
    
}

void OrientedBoundingBox::plotPlane(pcl::PointCloud<PointT>::Ptr cloud) {
    if (cloud->size() < 3) {
       ROS_ERROR("POINTS ON PLANE NOT PROVIDED");
       return;
    }

    for (int i = 0; i < 12; i += 3) {
       Eigen::Vector3f pt0 = cloud->points[i].getVector3fMap();
       Eigen::Vector3f pt2 = cloud->points[i+1].getVector3fMap() - pt0;
       Eigen::Vector3f pt1 = cloud->points[i+2].getVector3fMap() - pt0;
    
       Eigen::Vector3f normal = pt2.cross(pt1);

       std::cout << normal  << "\n";

       PointT color = cloud->points[i];
       for (float y = -1.0f; y < 1.0f; y += 0.01f) {
          for (float x = -1.0f; x < 1.0f; x += 0.01f) {
             PointT pt = color;
             pt.x = pt0(0) + pt1(0) * x + pt2(0) * y;
             pt.y = pt0(1) + pt1(1) * x + pt2(1) * y;
             pt.z = pt0(2) + pt1(2) * x + pt2(2) * y;
             cloud->push_back(pt);
          }
       }
    }
}

OrientedBoundingBox::PointT OrientedBoundingBox::Eigen2PointT(
    Eigen::Vector3f vec, Eigen::Vector3f color) {
    PointT pt;
    pt.x = vec(0); pt.r = color(0);
    pt.y = vec(1); pt.g = color(1);
    pt.z = vec(2); pt.b = color(2);
    return pt;
}
