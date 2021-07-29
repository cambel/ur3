cmake_minimum_required(VERSION 2.8.3)
project(ur_pykdl)

find_package(catkin
  REQUIRED
  COMPONENTS
  rospy
  geometry_msgs
  sensor_msgs
  tf
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
  rospy
  geometry_msgs
  sensor_msgs
  tf
  DEPENDS
)

install(
  DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
