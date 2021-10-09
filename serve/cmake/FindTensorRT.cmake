# This module defines the following variables:
#
# ::
#
# TensorRT_INCLUDE_DIRS TensorRT_LIBRARIES TensorRT_FOUND
#
# ::
#
# TensorRT_VERSION_STRING - version (x.y.z) TensorRT_VERSION_MAJOR  - major
# version (x) TensorRT_VERSION_MINOR  - minor version (y) TensorRT_VERSION_PATCH
# - patch version (z)
#
# Hints ^^^^^ A user may set ``TensorRT_ROOT`` to an installation root to tell
# this module where to look.
#

if(NOT TARGET CUDA::cudart)
  find_package(CUDAToolkit REQUIRED)
endif()

set(_TensorRT_SEARCHES)

if(TensorRT_ROOT)
  list(APPEND _TensorRT_SEARCHES ${TensorRT_ROOT} NO_DEFAULT_PATH)
endif()

# common paths
list(APPEND _TensorRT_SEARCHES /usr /usr/local/TensorRT)

# Include dir
find_path(
  TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ${_TensorRT_SEARCHES}
  PATH_SUFFIXES include)

# Nvinfer
find_library(
  TensorRT_NVINFER
  NAMES nvinfer
  HINTS ${_TensorRT_SEARCHES}
  PATH_SUFFIXES lib)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MAJOR
       REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MINOR
       REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_PATCH
       REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

  string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1"
                       TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1"
                       TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1"
                       TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
  set(TensorRT_VERSION_STRING
      "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}"
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TensorRT
  REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_NVINFER
  VERSION_VAR TensorRT_VERSION_STRING)

if(TensorRT_FOUND)
  set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
  get_filename_component(TensorRT_LIBRARY_DIR ${TensorRT_NVINFER} DIRECTORY
                         ABSOLUTE)

  function(_TensorRT_find_and_add_import_lib lib_name)
    find_library(
      TensorRT_${lib_name}
      NAMES ${lib_name}
      HINTS ${TensorRT_LIBRARY_DIR})

    mark_as_advanced(TensorRT_${lib_name})
    if(NOT TARGET TensorRT::${lib_name} AND TensorRT_${lib_name})
      add_library(TensorRT::${lib_name} UNKNOWN IMPORTED)
      target_include_directories(TensorRT::${lib_name} SYSTEM
                                 INTERFACE "${TensorRT_INCLUDE_DIRS}")
      target_link_libraries(TensorRT::${lib_name}
                            INTERFACE "${TensorRT_${lib_name}}")
      set_property(TARGET TensorRT::${lib_name}
                   PROPERTY IMPORTED_LOCATION "${TensorRT_${lib_name}}")
    endif()
  endfunction()
  foreach(trt_lib nvinfer nvinfer_plugin nvonnxparser)
    _tensorrt_find_and_add_import_lib(${trt_lib})
    _tensorrt_find_and_add_import_lib(${trt_lib}_static)
  endforeach()
endif()
