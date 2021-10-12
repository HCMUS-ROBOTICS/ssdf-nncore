#pragma once
#include <cuda_runtime_api.h>

#include <memory>

#include "serve/logger.hpp"

namespace ssdf::serve {
inline void cudaCheck(cudaError_t ret, const ILogger& logger) {
  if (ret != cudaSuccess) {
    logger.error("Cuda failure: {}", cudaGetErrorString(ret));
    abort();
  }
}

static auto StreamDeleter = [](cudaStream_t* stream_ptr) {
  if (stream_ptr) {
    cudaStreamDestroy(*stream_ptr);
    delete stream_ptr;
  }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream() {
  std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> stream_ptr(new cudaStream_t,
                                                                    StreamDeleter);
  if (cudaStreamCreateWithFlags(stream_ptr.get(), cudaStreamNonBlocking) != cudaSuccess) {
    stream_ptr.reset(nullptr);
  }
  return stream_ptr;
}
}  // namespace ssdf::serve
