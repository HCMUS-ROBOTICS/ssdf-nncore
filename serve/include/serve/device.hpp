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

static auto StreamDeleter = [](cudaStream_t* pStream) {
  if (pStream) {
    cudaStreamDestroy(*pStream);
    delete pStream;
  }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream() {
  std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
  if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess) {
    pStream.reset(nullptr);
  }
  return pStream;
}
}  // namespace ssdf::serve
