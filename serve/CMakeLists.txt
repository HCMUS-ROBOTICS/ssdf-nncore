cmake_minimum_required(VERSION 3.18)
project(serve LANGUAGES CUDA CXX)

find_package(catkin REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(TensorRT REQUIRED)

catkin_package(
  INCLUDE_DIRS
  include
  LIBRARIES
  ssdf_serve
  DEPENDS
  CUDAToolkit
  TensorRT)

add_subdirectory(src)
