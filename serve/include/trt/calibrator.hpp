#pragma once
#include <NvInfer.h>

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "serve/logger.hpp"

namespace ssdf::serve::trt {
class RndInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  RndInt8Calibrator(int batches, const std::filesystem::path& cache_file, const ILogger& logger,
                    const nvinfer1::INetworkDefinition& network, std::vector<int64_t>* elem_count);
  virtual ~RndInt8Calibrator();

  bool getBatch(void* bindings[], const char* names[], int nb_bingdinds) noexcept override;

  int getBatchSize() const noexcept override;

  const void* readCalibrationCache(size_t& length) noexcept override;

  void writeCalibrationCache(const void*, size_t) noexcept override {}

 private:
  int batches_;
  int current_batch_;
  std::filesystem::path cache_file_;
  std::unordered_map<std::string, void*> device_input_buffers_;
  std::vector<char> calibration_cache_;
  const ILogger& logger_;
};

}  // namespace ssdf::serve::trt
