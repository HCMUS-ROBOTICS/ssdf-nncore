#pragma once
#include <NvInfer.h>

#include <filesystem>
#include <vector>

#include "serve/logger.hpp"
#include "trt/option.hpp"

namespace ssdf::serve::trt {
bool broadcastIOFormats(const std::vector<IOFormat>& formats, size_t nb_bindings,
                        bool is_input = true);

template <typename A, typename B>
inline A divUp(A x, B n) {
  return (x + n - 1) / n;
}

uint32_t getElementSize(nvinfer1::DataType data_type) noexcept;

std::vector<char> loadTimingCacheFile(const std::filesystem::path& file_path,
                                      const ILogger& logger);

bool saveEngine(const nvinfer1::IHostMemory& serialized_engine,
                const std::filesystem::path& engine_path);

nvinfer1::Dims toDims(const std::vector<int>& vec, const ILogger& logger);

int64_t volume(const nvinfer1::Dims& d);
}  // namespace ssdf::serve::trt
