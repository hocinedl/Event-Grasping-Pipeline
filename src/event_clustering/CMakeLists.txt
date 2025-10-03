cmake_minimum_required(VERSION 3.0.2)
project(event_clustering)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries  message_generation
find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  cv_bridge
  dvs_msgs
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
  tf
  action_client_interface
  message_runtime
)

################################################
## Declare ROS messages, services and actions ##
################################################

## Tell CMake where to find your .action files
#add_action_files(
#  DIRECTORY action
#  FILES Clustering.action
#)

## Generate action message headers
#generate_messages(
#  DEPENDENCIES
#    actionlib_msgs
#    geometry_msgs
#)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS
    actionlib
    actionlib_msgs
    message_runtime
    cv_bridge
    dvs_msgs
    geometry_msgs
    sensor_msgs
    std_msgs
    tf
    action_client_interface
)

###########
## Build ##
###########
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
)

#############
## Install ##
#############
## Mark Python scripts for installation
catkin_install_python(PROGRAMS
  scripts/clustering_action_server2.py
  #scripts/clustering_action_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

