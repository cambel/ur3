cmake_minimum_required(VERSION 2.8.12)
project(ur3_kinematics)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  pluginlib
)
find_package(Boost REQUIRED COMPONENTS system)
find_package(PythonLibs 3 REQUIRED)

catkin_python_setup()

catkin_package(
  LIBRARIES
    kinematics
  CATKIN_DEPENDS
    pluginlib
  DEPENDS Boost
)

###########
## Build ##
###########

include_directories(SYSTEM ${Boost_INCLUDE_DIR})
include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(LAPACK REQUIRED)

set(LIB_KINEMATICSCPP ur3_kinematics)
set(PY_KINEMATICS arm)

set(LIB_UR3e ur3e_kinematics)
set(E_PY_KINEMATICS e_arm)

add_library(${LIB_KINEMATICSCPP}  src/kinematics.cpp )
add_library(${LIB_UR3e}  src/e_kinematics.cpp )

target_link_libraries(${LIB_KINEMATICSCPP} ${catkin_LIBRARIES} -llapack)
target_link_libraries(${LIB_UR3e} ${catkin_LIBRARIES} -llapack)

PYTHON_ADD_MODULE(${PY_KINEMATICS} src/pykinematics.cpp)
target_link_libraries(${PY_KINEMATICS} ${LIB_KINEMATICSCPP} boost_python boost_numpy ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} )
set_target_properties(${PY_KINEMATICS} PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})

PYTHON_ADD_MODULE(${E_PY_KINEMATICS} src/e_pykinematics.cpp)
target_link_libraries(${E_PY_KINEMATICS} ${LIB_UR3e} boost_python boost_numpy ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} )
set_target_properties(${E_PY_KINEMATICS} PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})

install(TARGETS ${PY_KINEMATICS} ${E_PY_KINEMATICS}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

install(TARGETS ${LIB_KINEMATICSCPP} ${LIB_UR3e} LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
