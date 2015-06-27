cmake_minimum_required(VERSION 2.8.3)
project(point_cloud_scene_decomposer)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  pcl_conversions
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

find_package(catkin REQUIRED COMPONENTS cmake_modules message_generation std_msgs std_srvs geometry_msgs roscpp rospy sensor_msgs visualization_msgs message_filters message_generation jsk_pcl_ros interactive_markers pcl_conversions jsk_topic_tools rviz eigen_conversions dynamic_tf_publisher jsk_interactive_marker jsk_recognition_msgs move_base_msgs rosgraph_msgs topic_tools jsk_topic_tools jsk_ik_server pcl_msgs jsk_footstep_msgs drc_com_common jsk_perception jsk_calibration resized_image_transport smach_msgs pcl_ros nav_msgs tf dynamic_reconfigure roseus)


## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

generate_dynamic_reconfigure_options(
    cfg/PointCloudFilterUtils.cfg)

## Generate messages in the 'msg' folder
add_message_files(
   FILES
   signal.msg
   #   Message2.msg
   )

## Generate services in the 'srv' folder
add_service_files(
  FILES
  ClusterVoxels.srv
#   Service2.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  #   sensor_msgs#
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
    INCLUDE_DIRS include
    LIBRARIES point_cloud_scene_decomposer
    #CATKIN_DEPENDS cv_bridge image_tranport pcl_conversions pcl_conversions roscpp rospy sensor_msgs std_msgs
    CATKIN_DEPENDS std_msgs
 DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  include ${catkin_INCLUDE_DIRS}
)

## Declare a cpp library
# add_library(point_cloud_scene_decomposer
#   src/${PROJECT_NAME}/point_cloud_scene_decomposer.cpp
# )

## Declare a cpp executable
add_executable(pcl_filter_utils_node
  src/pcl_filter_utils_node.cpp)
add_executable(point_cloud_image_creator_node
  src/point_cloud_image_creator_node.cpp)
add_executable(point_cloud_normal_gradient_node
  src/point_cloud_normal_gradient_node.cpp)
 add_executable(point_cloud_scene_decomposer_node 
   src/point_cloud_scene_decomposer_node.cpp
   src/scene_decomposer_image_processor.cpp
   src/region_adjacency_graph.cpp)
add_executable(point_cloud_cluster_matching_node
  src/point_cloud_cluster_matching_node.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
add_dependencies(pcl_filter_utils_node ${PROJECT_NAME}_gencfg)

## Specify libraries to link a library or executable target against
target_link_libraries(point_cloud_image_creator_node
  ${catkin_LIBRARIES})
target_link_libraries(pcl_filter_utils_node
  ${catkin_LIBRARIES})
target_link_libraries(point_cloud_normal_gradient_node
  ${catkin_LIBRARIES})
target_link_libraries(point_cloud_scene_decomposer_node
  ${catkin_LIBRARIES})
target_link_libraries(point_cloud_cluster_matching_node
   ${catkin_LIBRARIES})

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS point_cloud_scene_decomposer point_cloud_scene_decomposer_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_point_cloud_scene_decomposer.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
