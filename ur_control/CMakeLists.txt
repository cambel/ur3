cmake_minimum_required(VERSION 3.1.3)
project(ur_control)

find_package(catkin REQUIRED COMPONENTS
  control_msgs
  controller_manager
  effort_controllers
  gripper_action_controller
  joint_state_controller
  joint_trajectory_controller
  std_srvs
  geometry_msgs
  robot_state_publisher
  sensor_msgs
  trajectory_msgs
  trac_ik_python
)

catkin_python_setup()

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})
