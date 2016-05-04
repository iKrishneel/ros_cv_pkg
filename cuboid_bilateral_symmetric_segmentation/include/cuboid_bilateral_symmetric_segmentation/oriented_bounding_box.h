
#ifndef _ORIENTED_BOUNDING_BOX_H_
#define _ORIENTED_BOUNDING_BOX_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/correspondence.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/distances.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/common/pca.h>

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <jsk_recognition_msgs/ModelCoefficientsArray.h>
#include <jsk_recognition_utils/pcl_util.h>
#include <jsk_recognition_utils/geo_util.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>

namespace jsk_msgs = jsk_recognition_msgs;

class OrientedBoundingBox {
 private:  
    typedef pcl::PointXYZRGB PointT;
    pcl::ExtractIndices<PointT> extract;
    bool computeBoundingBox(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr,
			    const Eigen::Vector4f,
			    const jsk_msgs::PolygonArrayConstPtr &,
			    const jsk_msgs::ModelCoefficientsArrayConstPtr &,
			    jsk_msgs::BoundingBox &);
    int findNearestPlane(const Eigen::Vector4f &,
			 const jsk_msgs::PolygonArrayConstPtr &,
			 const jsk_msgs::ModelCoefficientsArrayConstPtr &);
    bool align_boxes_;
    bool use_pca_;
    bool force_to_flip_z_axis_;
   
 public:
    OrientedBoundingBox();
    bool fitOriented3DBoundingBox(jsk_msgs::BoundingBox &,
				  const pcl::PointCloud<PointT>::Ptr,
				  const jsk_msgs::PolygonArrayConstPtr &,
				  const jsk_msgs::ModelCoefficientsArrayConstPtr &);
};


#endif  // _ORIENTED_BOUNDING_BOX_H_
