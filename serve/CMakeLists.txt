cmake_minimum_required(VERSION 3.18)
project(serve LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(TensorRT REQUIRED)

add_subdirectory(src)
