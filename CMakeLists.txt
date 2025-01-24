cmake_minimum_required(VERSION 3.0.2)

project(human_articular_space) # Project name

## Find the necessary ROS packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs # Para el nodo conversor
  skeleton_3d # This package contains the Skeleton3D message
  message_generation
  urdf
)

## Specify the message files to be generated
add_message_files(
  FILES

  # From IK_arm_calculator.py
  HumanBody.msg
  DHAnglesArm.msg

  # From tf2_listener.py
  KP_URDF.msg
  KP_vision.msg # DEBUG only
)

## Generate messages and services with the listed dependencies
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs # Para el nodo conversor
  skeleton_3d # This package contains the Skeleton3D message
)

## Define the package and its dependencies
catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    sensor_msgs # Para el nodo conversor
    skeleton_3d # This package contains the Skeleton3D message
    message_runtime
)

## Include directories for the project
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Install Python executable scripts
install(PROGRAMS
  scripts/IK_arm_calculator.py
  scripts/human_body_to_joint_states_converter.py
  scripts/pose_shoulder_TF_publisher.py
  scripts/tf2_listener.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install additional resources (launch files, configs, RViz settings)
install(DIRECTORY 
  launch 
  config
  rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
# RCO: mirar que hace esto
## Optional: Add environment hooks to reload configurations
# catkin_add_env_hooks(99.skeleton_3d
#  SHELLS sh
#  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks
# )

## Display a status message once the CMake configuration completes
message(STATUS "CMake configuration for human_articular_space completed successfully.")
