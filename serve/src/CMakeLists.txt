add_library(ssdf_serve trt/calibrator.cpp trt/generator.cpp trt/utils.cpp)
target_include_directories(ssdf_serve PUBLIC ${PROJECT_SOURCE_DIR}/include
                                             ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(ssdf_serve CUDA::cudart fmt::fmt TensorRT::nvonnxparser
                      TensorRT::nvinfer)
target_compile_features(ssdf_serve PRIVATE cxx_std_17 cuda_std_17)
