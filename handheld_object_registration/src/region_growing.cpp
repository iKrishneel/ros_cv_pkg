
#include <handheld_object_registration/region_growing.h>

RegionGrowing::RegionGrowing(
    const sensor_msgs::CameraInfo::ConstPtr camera_info,
    const int threads) :
    num_threads_(threads) {
    this->camera_info_ = camera_info;
    this->kdtree_ = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
}

void RegionGrowing::fastSeedRegionGrowing(
    pcl::PointCloud<PointNormalT>::Ptr src_points,
    cv::Point2i &seed_index2D, const PointCloud::Ptr cloud,
    const PointNormal::Ptr normals, const PointT seed_pt) {
    if (cloud->empty() || normals->size() != cloud->size()) {
       return;
    }
    cv::Point2f image_index;
    int seed_index = -1;
    if (this->projectPoint3DTo2DIndex(image_index, seed_pt)) {
       seed_index = (static_cast<int>(image_index.x) +
                     (static_cast<int>(image_index.y) * camera_info_->width));
       seed_index2D = cv::Point2i(static_cast<int>(image_index.x),
                                   static_cast<int>(image_index.y));
    } else {
       ROS_ERROR("INDEX IS NAN");
       return;
    }

#ifdef _DEBUG
    cv::Mat test = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::circle(test, image_index, 3, cv::Scalar(0, 255, 0), -1);
    cv::imshow("test", test);
    cv::waitKey(3);
#endif
    
    Eigen::Vector4f seed_point = cloud->points[seed_index].getVector4fMap();
    Eigen::Vector4f seed_normal = normals->points[
       seed_index].getNormalVector4fMap();
    
    std::vector<int> processing_list;
    std::vector<int> labels(static_cast<int>(cloud->size()), -1);

    const int window_size = 3;
    const int wsize = window_size * window_size;
    const int lenght = std::floor(window_size/2);

    processing_list.clear();
    for (int j = -lenght; j <= lenght; j++) {
       for (int i = -lenght; i <= lenght; i++) {
          int index = (seed_index + (j * camera_info_->width)) + i;
          if (index >= 0 && index < cloud->size()) {
             processing_list.push_back(index);
          }
       }
    }
    std::vector<int> temp_list;
    while (true) {
       if (processing_list.empty()) {
          break;
       }
       temp_list.clear();
       for (int i = 0; i < processing_list.size(); i++) {
          int idx = processing_list[i];
          if (labels[idx] == -1) {
             Eigen::Vector4f c = cloud->points[idx].getVector4fMap();
             Eigen::Vector4f n = normals->points[idx].getNormalVector4fMap();
             
             if (this->seedVoxelConvexityCriteria(
                    seed_point, seed_normal, seed_point, c, n, -0.01) == 1) {
                labels[idx] = 1;

                for (int j = -lenght; j <= lenght; j++) {
                   for (int k = -lenght; k <= lenght; k++) {
                      int index = (idx + (j * camera_info_->width)) + k;
                      if (index >= 0 && index < cloud->size()) {
                         temp_list.push_back(index);
                      }
                   }
                }
             }
          }
       }
       processing_list.clear();
       processing_list.insert(processing_list.end(), temp_list.begin(),
                              temp_list.end());
    }
    src_points->clear();
    for (int i = 0; i < labels.size(); i++) {
       if (labels[i] != -1) {
          PointNormalT pt;
          pt.x = cloud->points[i].x;
          pt.y = cloud->points[i].y;
          pt.z = cloud->points[i].z;
          pt.r = cloud->points[i].r;
          pt.g = cloud->points[i].g;
          pt.b = cloud->points[i].b;
          pt.normal_x = normals->points[i].normal_x;
          pt.normal_y = normals->points[i].normal_y;
          pt.normal_z = normals->points[i].normal_z;
          src_points->push_back(pt);
       }
    }
}

bool RegionGrowing::seedRegionGrowing(
    pcl::PointCloud<PointNormalT>::Ptr src_points,
    const PointT seed_point, const PointCloud::Ptr cloud,
    PointNormal::Ptr normals) {
    if (cloud->empty() || normals->size() != cloud->size()) {
       ROS_ERROR("- Region growing failed. Incorrect inputs sizes ");
       return false;
    }
    if (isnan(seed_point.x) || isnan(seed_point.y) || isnan(seed_point.z)) {
       ROS_ERROR("- Seed Point is Nan. Skipping");
       return false;
    }

    this->kdtree_->setInputCloud(cloud);
    
    std::vector<int> neigbor_indices;
    this->getPointNeigbour<int>(neigbor_indices, seed_point, 1);
    int seed_index = neigbor_indices[0];
    
    const int in_dim = static_cast<int>(cloud->size());
    int *labels = reinterpret_cast<int*>(malloc(sizeof(int) * in_dim));
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_)
#endif
    for (int i = 0; i < in_dim; i++) {
       if (i == seed_index) {
          labels[i] = 1;
       }
       labels[i] = -1;
    }
    this->seedCorrespondingRegion(labels, cloud, normals,
                                  seed_index, seed_index);
    src_points->clear();
    for (int i = 0; i < in_dim; i++) {
       if (labels[i] != -1) {
          PointNormalT pt;
          pt.x = cloud->points[i].x;
          pt.y = cloud->points[i].y;
          pt.z = cloud->points[i].z;
          pt.r = cloud->points[i].r;
          pt.g = cloud->points[i].g;
          pt.b = cloud->points[i].b;
          pt.normal_x = normals->points[i].normal_x;
          pt.normal_y = normals->points[i].normal_y;
          pt.normal_z = normals->points[i].normal_z;
          src_points->push_back(pt);
       }
    }
    free(labels);
    return true;
}

void RegionGrowing::seedCorrespondingRegion(
    int *labels, const PointCloud::Ptr cloud, const PointNormal::Ptr normals,
    const int parent_index, const int seed_index) {
    Eigen::Vector4f seed_point = cloud->points[seed_index].getVector4fMap();
    Eigen::Vector4f seed_normal = normals->points[
       seed_index].getNormalVector4fMap();
    
    std::vector<int> neigbor_indices;
    this->getPointNeigbour<int>(neigbor_indices,
                                cloud->points[parent_index], 18);

    int neigb_lenght = static_cast<int>(neigbor_indices.size());
    std::vector<int> merge_list(neigb_lenght);
    merge_list[0] = -1;
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_)
#endif
    for (int i = 1; i < neigbor_indices.size(); i++) {
        int index = neigbor_indices[i];
        if (index != parent_index && labels[index] == -1) {
            Eigen::Vector4f parent_pt = cloud->points[
                parent_index].getVector4fMap();
            Eigen::Vector4f child_pt = cloud->points[index].getVector4fMap();
            Eigen::Vector4f child_norm = normals->points[
                index].getNormalVector4fMap();
            if (this->seedVoxelConvexityCriteria(
                   seed_point, seed_normal, parent_pt, child_pt,
                   child_norm, -0.01f) == 1) {
                merge_list[i] = index;
                labels[index] = 1;
            } else {
                merge_list[i] = -1;
            }
        } else {
            merge_list[i] = -1;
        }
    }
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_) schedule(guided, 1)
#endif
    for (int i = 0; i < merge_list.size(); i++) {
        int index = merge_list[i];
        if (index != -1) {
           seedCorrespondingRegion(labels, cloud, normals, index, seed_index);
        }
    }
}

int RegionGrowing::seedVoxelConvexityCriteria(
    Eigen::Vector4f seed_point, Eigen::Vector4f seed_normal,
    Eigen::Vector4f c_centroid, Eigen::Vector4f n_centroid,
    Eigen::Vector4f n_normal, const float thresh) {
    // float im_relation = (n_centroid - c_centroid).dot(n_normal);
    float pt2seed_relation = FLT_MAX;
    float seed2pt_relation = FLT_MAX;
    pt2seed_relation = (n_centroid - seed_point).dot(n_normal);
    seed2pt_relation = (seed_point - n_centroid).dot(seed_normal);
    if (seed2pt_relation > thresh && pt2seed_relation > thresh) {
      return 1;
    } else {
       return -1;
    }
}
