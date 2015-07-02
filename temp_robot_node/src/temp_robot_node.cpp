// Copyright (C) 2015 by Krishneel Chaudhary, JSK

#include <temp_robot_node/temp_robot_node.h>

RobotNode::RobotNode() :
    processing_counter_(0) {
   
    this->subsribe();
    this->onInit();
}

void RobotNode::onInit() {
    pub_cloud_ = pnh_.advertise<
       point_cloud_scene_decomposer::signal>(
          "/robot_pushing_motion_node/output/signal", 1);
}

void RobotNode::subsribe() {
    this->sub_cloud_ = this->pnh_.subscribe(
        "input_signal", 1, &RobotNode::signalCallback, this);
}

void RobotNode::signalCallback(
    const point_cloud_scene_decomposer::signal &signal_msg) {
    ROS_INFO("-- PROCESSING ROBOT NODE...");

    if (signal_msg.command == 2 &&
        signal_msg.counter == processing_counter_) {
       ros::Duration(15).sleep();
       processing_counter_++;
    }
    point_cloud_scene_decomposer::signal signal;
    signal.header = signal_msg.header;
    signal.command = 3;
    signal.counter = this->processing_counter_ - 1;
    this->pub_cloud_.publish(signal);

    std::cout << "Signal Value: " << signal.command << "\t" <<
              signal.counter << std::endl;
}

int main(int argc, char *argv[]) {

    ros::init(argc, argv, "temp_robot_node");
    RobotNode pcfu;
    ros::spin();
    return 0;
}
