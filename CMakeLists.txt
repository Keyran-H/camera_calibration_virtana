cmake_minimum_required(VERSION 2.8.3)
project(camera_calibration_virtana)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  tf2
  tf
  gazebo_msgs
  dynamic_reconfigure
  cv_bridge
  image_transport
)

find_package(Ceres REQUIRED PATHS "/usr/local/lib/cmake/Ceres/")

find_package(OpenCV REQUIRED PATHS "opt/ros/kinetic/share/OpenCV-3.3.1-dev/")


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

generate_dynamic_reconfigure_options(
  cfg/calibration.cfg
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES camera_calibration_virtana
)

add_executable(nlo_eg src/nlo_eg.cpp)
add_executable(nlo_eg_cp src/nlo_camera_and_point.cpp)
add_executable(world_transform_pub src/world_transform_pub.cpp)
add_executable(image_proc_node src/image_proc_node.cpp)

add_dependencies(world_transform_pub ${PROJECT_NAME}_gencfg)


add_library(camera_calibration_virtana
   src/camera_calibration_virtana/world_transform.cpp
   src/camera_calibration_virtana/image_proc.cpp
)


target_link_libraries(nlo_eg
   ${catkin_LIBRARIES}
   ${CERES_LIBRARIES}
   ${OpenCV_LIBRARIES}
)

target_link_libraries(nlo_eg_cp
   ${catkin_LIBRARIES}
   ${CERES_LIBRARIES}
   ${OpenCV_LIBRARIES}
)

target_link_libraries(camera_calibration_virtana
   ${CERES_LIBRARIES}
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
)

target_link_libraries(world_transform_pub
   camera_calibration_virtana
)

target_link_libraries(image_proc_node
   camera_calibration_virtana
)

install(DIRECTORY include/camera_calibration_virtana/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(TARGETS
  camera_calibration_virtana
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
