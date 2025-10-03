cmake_minimum_required(VERSION 3.0.2)
project(action_client_interface)

find_package(catkin REQUIRED COMPONENTS
  actionlib_msgs
  geometry_msgs
  std_msgs
  message_generation
)

## Declare the action file
add_action_files(
  DIRECTORY action
  FILES Clustering.action
)

## Generate the messages
generate_messages(
  DEPENDENCIES actionlib_msgs geometry_msgs std_msgs
)

## Export the generated messages
catkin_package(
  CATKIN_DEPENDS actionlib_msgs geometry_msgs std_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

