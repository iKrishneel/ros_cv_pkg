
#include <collision_check_grasp_plannar/collision_check_grasp_plannar.h>

CollisionCheckGraspPlannar::CollisionCheckGraspPlannar() {
   
}

void CollisionCheckGraspPlannar::onInit() {
   
}

void CollisionCheckGraspPlannar::subscribe() {
   
}

void CollisionCheckGraspPlannar::unsubscribe() {
    this->sub_boxes_.unsubscribe();
    this->sub_cloud_.unsubscribe();
}

void CollisionCheckGraspPlannar::cloudCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const jsk_msgs::BoundingBoxArrayConstPtr &boxes_msg) {
   
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "collision_check_grasp_plannar");
    CollisionCheckGraspPlannar ccgp;
    ros::spin();
    return 0;
}
