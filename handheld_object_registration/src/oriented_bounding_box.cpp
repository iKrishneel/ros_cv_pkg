
#include <handheld_object_registration/oriented_bounding_box.h>

OrientedBoundingBox::OrientedBoundingBox() :
    num_points_(3), num_planes_(4), use_pca_(true),
    force_to_flip_z_axis_(true), align_boxes_(true) {
    this->colors_.push_back(Eigen::Vector3f(255, 0, 0));  // D2
    this->colors_.push_back(Eigen::Vector3f(0, 255, 0));  // D1
    this->colors_.push_back(Eigen::Vector3f(0, 0, 255));  // C1
    this->colors_.push_back(Eigen::Vector3f(255, 0, 255));  // C2
}

bool OrientedBoundingBox::fitOriented3DBoundingBox(
    jsk_msgs::BoundingBox &bounding_box,
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud) {
    if (in_cloud->empty()) {
       ROS_ERROR("EMPTY DATA FOR FITTING BOUNDING BOX");
       return false;
    }

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<pcl::PointXYZRGBNormal, PointT>(
       *in_cloud, *cloud);
    
    pcl_msgs::ModelCoefficients coeff;
    coeff.values.resize(4);
    coeff.values[0] = -0.013253158889710903;
    coeff.values[1] = -0.7634825110435486;
    coeff.values[2] = -0.6456924676895142;
    coeff.values[3] = 0.5587531924247742;

    jsk_msgs::ModelCoefficientsArrayPtr coefficients(
       new jsk_msgs::ModelCoefficientsArray);
    coefficients->coefficients.push_back(coeff);
    
    geometry_msgs::Polygon polygon;
    geometry_msgs::Point32 points;
    points.x = -0.295990467072;
    points.y = -0.295990467072;
    points.z = -0.295990467072;
    polygon.points.push_back(points);
    points.x = -0.403368830681;
    points.y = 0.131219238043;
    points.z = 0.718341171741;
    polygon.points.push_back(points);
    points.x = -0.265710562468;
    points.y = 0.239143759012;
    points.z = 0.587907135487;
    polygon.points.push_back(points);
    points.x = -0.25840857625;
    points.y = 0.241782158613;
    points.z = 0.584642469883;
    polygon.points.push_back(points);
    points.x = -0.25694078207;
    points.y = 0.24209433794;
    points.z = 0.584244549274;
    polygon.points.push_back(points);
    points.x = -0.253921985626;
    points.y = 0.242065608501;
    points.z = 0.584220290184;
    polygon.points.push_back(points);
    points.x = -0.252915769815;
    points.y = 0.242056027055;
    points.z = 0.584212183952;
    polygon.points.push_back(points);
    points.x = -0.0429002977908;
    points.y = 0.236454889178;
    points.z = 0.586788535118;
    polygon.points.push_back(points);
    points.x = -0.0378134213388;
    points.y = 0.236082538962;
    points.z = 0.587131202221;
    polygon.points.push_back(points);
    points.x = -0.0368532538414;
    points.y = 0.235749393702;
    points.z = 0.587507009506;
    polygon.points.push_back(points);
    points.x = -0.0363084413111;
    points.y = 0.225820317864;
    points.z = 0.599251925945;
    polygon.points.push_back(points);
    points.x = -0.0362884178758;
    points.y = 0.224944084883;
    points.z = 0.60028898716;
    polygon.points.push_back(points);
    points.x = -0.0399563014507;
    points.y = 0.205955728889;
    points.z = 0.622840821743;
    polygon.points.push_back(points);
    points.x = -0.045963421464;
    points.y = 0.186520799994;
    points.z = 0.645966589451;
    polygon.points.push_back(points);
    points.x = -0.0610990040004;
    points.y = 0.137700140476;
    points.z = 0.704059541225;
    polygon.points.push_back(points);
    points.x = -0.0632578730583;
    points.y = 0.130998879671;
    points.z = 0.712035059929;
    polygon.points.push_back(points);
    points.x = -0.0915021151304;
    points.y = 0.0885300114751;
    points.z = 0.762861013412;
    polygon.points.push_back(points);
    points.x = -0.0953296869993;
    points.y = 0.0828716158867;
    points.z = 0.769634127617;
    polygon.points.push_back(points);
    points.x = -0.171860456467;
    points.y = 0.0139996558428;
    points.z = 0.852652072906;
    polygon.points.push_back(points);
    points.x = -0.175178423524;
    points.y = 0.0136462403461;
    points.z = 0.853134453297;
    polygon.points.push_back(points);
    points.x = -0.500306665897;
    points.y = -0.0195564702153;
    points.z = 0.898722171783;
    polygon.points.push_back(points);
    points.x = -0.501877427101;
    points.y = -0.0195415355265;
    points.z = 0.898734748363;
    polygon.points.push_back(points);
    points.x = -0.501892387867;
    points.y = -0.0188872888684;
    points.z = 0.897960484028;
    polygon.points.push_back(points);
    points.x = -0.501907348633;
    points.y = -0.0182329937816;
    points.z = 0.897186160088;
    polygon.points.push_back(points);
    points.x = -0.492967247963;
    points.y = -0.00386042939499;
    points.z = 0.879997372627;
    polygon.points.push_back(points);
    points.x = -0.490170508623;
    points.y = 0.000503135146573;
    points.z = 0.874777078629;
    polygon.points.push_back(points);
    points.x = -0.473379850388;
    points.y = 0.0262248814106;
    points.z = 0.843999862671;
    polygon.points.push_back(points);
    points.x = -0.458818823099;
    points.y = 0.0481183938682;
    points.z = 0.817798137665;
    polygon.points.push_back(points);
    
    geometry_msgs::PolygonStamped poly_stamped;
    poly_stamped.polygon = polygon;
    jsk_msgs::PolygonArrayPtr planes(new jsk_msgs::PolygonArray);
    planes->polygons.push_back(poly_stamped);

    Eigen::Vector4f center;
    pcl::compute3DCentroid(*cloud, center);
    bool successp = computeBoundingBox(cloud, center, planes,
                                       coefficients, bounding_box);
    if (!successp) {
       return false;
    }
    return true;
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
    std::vector<Eigen::Vector4f> &plane_coefficients,
    pcl::PointCloud<PointT>::Ptr cloud,
    const jsk_msgs::BoundingBox bounding_box, const bool is_plot) {
    Eigen::Vector3f center = Eigen::Vector3f(bounding_box.pose.position.x,
                                             bounding_box.pose.position.y,
                                             bounding_box.pose.position.z);
    Eigen::Vector3f dims = Eigen::Vector3f(bounding_box.dimensions.x / 2.0f,
                                           bounding_box.dimensions.y / 2.0f,
                                           bounding_box.dimensions.z / 2.0f);
    
    std::vector<std::vector<Eigen::Vector3f> > corners(this->num_planes_);
    std::vector<Eigen::Vector3f> corner(this->num_points_);
    // 4 corners
    /*
    corner[0] = Eigen::Vector3f(dims(0), dims(1), dims(2));
    corner[1] = Eigen::Vector3f(-dims(0), -dims(1), -dims(2));
    corner[2] = Eigen::Vector3f(-dims(0), -dims(1), dims(2));
    corners[0] = corner;
    
    std::vector<Eigen::Vector3f> corner2(this->num_points_);
    corner2[0] = Eigen::Vector3f(dims(0), -dims(1), dims(2));
    corner2[1] = Eigen::Vector3f(-dims(0), dims(1), -dims(2));
    corner2[2] = Eigen::Vector3f(-dims(0), dims(1), dims(2));
    corners[1] = corner2;
    */
    // 4 mid-points
    std::vector<Eigen::Vector3f> corner3(this->num_points_);
    corner3[0] = Eigen::Vector3f(dims(0), 0, -dims(2));
    corner3[1] = Eigen::Vector3f(-dims(0), 0, -dims(2));
    corner3[2] = Eigen::Vector3f(-dims(0), 0, dims(2));
    corners[2] = corner3;

    std::vector<Eigen::Vector3f> corner4(this->num_points_);
    corner4[0] = Eigen::Vector3f(0, -dims(1), -dims(2));
    corner4[1] = Eigen::Vector3f(0, dims(1), dims(2));
    corner4[2] = Eigen::Vector3f(0, dims(1), -dims(2));
    corners[3] = corner4;
    
    cloud->clear();
    for (int i = 0; i < corners.size(); i++) {
       for (int j = 0; j < corners[i].size(); j++) {
          PointT pt = this->Eigen2PointT(corners[i][j], colors_[i]);
          cloud->push_back(pt);
       }
    }
    // Trasformation
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

    this->computePlaneCoefficients(plane_coefficients, cloud);

    if (is_plot) {
       pcl::PointCloud<PointT>::Ptr plane_cloud(new pcl::PointCloud<PointT>);
       PointT pt = this->Eigen2PointT(center, Eigen::Vector3f(255, 255, 255));
       // cloud->push_back(pt);
       this->plotPlane(plane_cloud, cloud);
       cloud->clear();
       *cloud = *plane_cloud;
    }
}

bool OrientedBoundingBox::computePlaneCoefficients(
    std::vector<Eigen::Vector4f> &plane_coefficients,
    const pcl::PointCloud<PointT>::Ptr plane_points) {
    if (plane_points->size() < this->num_points_) {
       ROS_ERROR("TO FEW POINTS \nCANNOT COMPUTE PLANE COEFFICIENTS");
       return false;
    }
    for (int i = 0; i < plane_points->size(); i += this->num_points_) {
       Eigen::Vector3f pt0 = plane_points->points[i].getVector3fMap();
       Eigen::Vector3f pt2 = plane_points->points[i+2].getVector3fMap() - pt0;
       Eigen::Vector3f pt1 = plane_points->points[i+1].getVector3fMap() - pt0;
       Eigen::Vector3f normal = pt2.cross(pt1);
       if (!isnan(normal(0)) && !isnan(normal(1)) && !isnan(normal(2))) {
          Eigen::Vector4f coef;
          coef.head<3>() = normal;
          coef(3) = (normal(0) * pt0(0)) +
             (normal(1) * pt0(1)) + (normal(2) * pt0(2));
          plane_coefficients.push_back(coef);
       }
    }
    return true;
}

void OrientedBoundingBox::plotPlane(
    pcl::PointCloud<PointT>::Ptr cloud,
    const pcl::PointCloud<PointT>::Ptr plane_points,
    const int s_index, const int t_plane) {
    if (plane_points->size() < this->num_points_) {
       ROS_ERROR("POINTS ON PLANE NOT PROVIDED");
       return;
    }
    // cloud->clear();
    for (int i = s_index; i < /*plane_points->size() -*/ t_plane;
         i += this->num_points_) {
       Eigen::Vector3f pt0 = plane_points->points[i].getVector3fMap();
       Eigen::Vector3f pt2 = plane_points->points[i+2].getVector3fMap() - pt0;
       Eigen::Vector3f pt1 = plane_points->points[i+1].getVector3fMap() - pt0;
       Eigen::Vector3f normal = pt2.cross(pt1);
       
       PointT color = plane_points->points[i];
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

void OrientedBoundingBox::plotPlane(
    pcl::PointCloud<PointT>::Ptr cloud, const Eigen::Vector4f param,
    const Eigen::Vector3f color) {
    Eigen::Vector3f center = Eigen::Vector3f(param(3)/param(0), 0, 0);
    Eigen::Vector3f normal = param.head<3>();
    float coef = normal.dot(center);
    float x = coef / normal(0);
    float y = coef / normal(1);
    float z = coef / normal(2);
    Eigen::Vector3f point_x = Eigen::Vector3f(x, 0.0f, 0.0f);
    Eigen::Vector3f point_y = Eigen::Vector3f(0.0f, y, 0.0f) - point_x;
    Eigen::Vector3f point_z = Eigen::Vector3f(0.0f, 0.0f, z) - point_x;
    for (float y = -1.0f; y < 1.0f; y += 0.01f) {
       for (float x = -1.0f; x < 1.0f; x += 0.01f) {
          PointT pt;
          pt.x = point_x(0) + point_y(0) * x + point_z(0) * y;
          pt.y = point_x(1) + point_y(1) * x + point_z(1) * y;
          pt.z = point_x(2) + point_y(2) * x + point_z(2) * y;
          pt.g = 255;
          cloud->push_back(pt);
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
