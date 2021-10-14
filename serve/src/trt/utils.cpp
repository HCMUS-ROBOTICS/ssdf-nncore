#include "trt/utils.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>

namespace ssdf::serve::trt {
bool broadcastIOFormats(const std::vector<IOFormat>& formats, size_t nb_bindings, bool is_input) {
  bool broadcast = formats.size() == 1,
       valid_formats_count = broadcast || (formats.size() == nb_bindings);
  if (!formats.empty() && !valid_formats_count) {
    throw std::invalid_argument(is_input ? "The number of inputIOFormats must match network's "
                                           "inputs or be one for broadcasting."
                                         : "The number of outputIOFormats must match network's "
                                           "outputs or be one for broadcasting.");
  }
  return broadcast;
}

uint32_t getElementSize(nvinfer1::DataType data_type) noexcept {
  switch (data_type) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  return 0;
}

std::vector<char> loadTimingCacheFile(const std::filesystem::path& file_path,
                                      const ILogger& logger) {
  std::ifstream iFile(file_path, std::ios::in | std::ios::binary);
  if (!iFile) {
    // TODO(all) remove c_str() when std::format is supported
    logger.warn(
        "Could not read timing cache from: {}. A new timing cache will be generated and written.",
        file_path.c_str());
    return {};
  }
  size_t file_size = std::filesystem::file_size(file_path);
  std::vector<char> content(file_size);
  iFile.read(content.data(), file_size);
  logger.info("Loaded {} bytes of timing cache from {}", file_size, file_path.c_str());
  return content;
}

bool saveEngine(const nvinfer1::IHostMemory& serialized_engine,
                const std::filesystem::path& engine_path) {
  std::ofstream engine_file(engine_path, std::ios::binary);
  if (!engine_file) {
    return false;
  }
  engine_file.write(static_cast<char*>(serialized_engine.data()), serialized_engine.size());
  return !engine_file.fail();
}

nvinfer1::Dims toDims(const std::vector<int>& vec, const ILogger& logger) {
  int limit = static_cast<int>(nvinfer1::Dims::MAX_DIMS);
  if (static_cast<int>(vec.size()) > limit) {
    logger.warn("Vector too long, only first 8 elements are used in dimension.");
  }
  // Pick first nvinfer1::Dims::MAX_DIMS elements
  nvinfer1::Dims dims{std::min(static_cast<int>(vec.size()), limit), {}};
  std::copy_n(vec.begin(), dims.nbDims, std::begin(dims.d));
  return dims;
}

int64_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}
}  // namespace ssdf::serve::trt
