cmake_minimum_required(VERSION 2.4.6)
# include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# rosbuild_init()

# IF(NOT ROS_BUILD_TYPE)
#   SET(ROS_BUILD_TYPE Release)
# ENDIF()

# MESSAGE("Build type: " ${ROS_BUILD_TYPE})

# ROS2 必须
find_package(ament_cmake REQUIRED)

# ROS2 packages
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

# Check c++17 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX17)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
   add_definitions(-DCOMPILEDWITHC17)
   message(STATUS "Using flag -std=c++17.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no c++17 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake_modules)

# 要保证整个工程的opencv版本一致，包括dbow，源码以及ros相关的
# 3 4 都可以正常运行
find_package(OpenCV 3.2)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 3.2 not found.")
   endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../../
${PROJECT_SOURCE_DIR}/../../../include
${PROJECT_SOURCE_DIR}/../../../include/CameraModels
${PROJECT_SOURCE_DIR}/../../../Thirdparty/Sophus
${Pangolin_INCLUDE_DIRS}
)

set(LIBS 
${OpenCV_LIBS} 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/../../../Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/../../../Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/../../../lib/libORB_SLAM3.so
-lboost_system
)

# # Node for monocular camera
# rosbuild_add_executable(Mono
# src/ros_mono.cc
# )

# target_link_libraries(Mono
# ${LIBS}
# )

# Node for monocular camera (Augmented Reality Demo)
# 由于里面代码没有将cv::mat 改成 SOPUS的暂时注释掉，不然会报错
# rosbuild_add_executable(MonoAR
# src/AR/ros_mono_ar.cc
# src/AR/ViewerAR.h
# src/AR/ViewerAR.cc
# )

# target_link_libraries(MonoAR
# ${LIBS}
# )

# # Node for stereo camera
# rosbuild_add_executable(Stereo
# src/ros_stereo.cc
# )

# target_link_libraries(Stereo
# ${LIBS}
# )

# # Node for RGB-D camera
# rosbuild_add_executable(RGBD
# src/ros_rgbd.cc
# )

# target_link_libraries(RGBD
# ${LIBS}
# )

# # Node for monocular-inertial camera
# rosbuild_add_executable(Mono_Inertial
# src/ros_mono_inertial.cc
# )

# target_link_libraries(Mono_Inertial
# ${LIBS}
# )

# # Node for stereo-inertial camera
# rosbuild_add_executable(Stereo_Inertial
# src/ros_stereo_inertial.cc
# )

# target_link_libraries(Stereo_Inertial
# ${LIBS}
# )
add_executable(ros_rgbd_ros2 src/ros_rgbd_ros2.cc)
ament_target_dependencies(ros_rgbd_ros2
rclcpp
sensor_msgs
td_msgs
cv_bridge
image_transport
message_filters 
)

target_link_libraries(ros_rgbd_ros2
${LIBS}
)