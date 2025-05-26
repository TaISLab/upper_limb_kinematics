cmake_minimum_required(VERSION 3.0.2)

project(upper_limb_kinematics) # Project name

catkin_python_setup() # Para ejecutar el setup.py de la raiz para ubicar los módulos. 

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

  RightArm.msg  # From IK_arm_calculator.py
  KP_URDF.msg   # From DK_listener.py
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
  scripts/IK_arm.py
  scripts/DK_listener.py
  scripts/TF_shoulder_publisher.py
  scripts/compute_confort_index.py
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
