cmake_minimum_required(VERSION 2.8.12)
project(ur3_ikfast_plugin)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  moveit_core
  pluginlib
  roscpp
  tf_conversions
)

include_directories(${catkin_INCLUDE_DIRS})

catkin_package(
  LIBRARIES
  CATKIN_DEPENDS
    moveit_core
    pluginlib
    roscpp
    tf_conversions
)

include_directories(include)

set(IKFAST_LIBRARY_NAME_ARM ur3_arm_ikfast_moveit_plugin)
set(IKFAST_LIBRARY_NAME_EXT_EE ur3_ext_ee_ikfast_moveit_plugin)

find_package(LAPACK REQUIRED)

add_library(${IKFAST_LIBRARY_NAME_ARM} src/ur3_arm_ikfast_moveit_plugin.cpp)
add_library(${IKFAST_LIBRARY_NAME_EXT_EE} src/ur3_ext_ee_ikfast_moveit_plugin.cpp)

target_link_libraries(${IKFAST_LIBRARY_NAME_ARM} ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${LAPACK_LIBRARIES})
target_link_libraries(${IKFAST_LIBRARY_NAME_EXT_EE} ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${LAPACK_LIBRARIES})

install(TARGETS ${IKFAST_LIBRARY_NAME_ARM} ${IKFAST_LIBRARY_NAME_EXT_EE} LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

install(
  FILES
  ur3_arm_description.xml
  ur3_extended_ee_description.xml
  DESTINATION
  ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
