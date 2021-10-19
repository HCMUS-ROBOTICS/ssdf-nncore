#pragma once
#include <NvInfer.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "serve/inference.hpp"
#include "serve/logger.hpp"

namespace ssdf::serve::trt {
class TRTBackend : public IBackend {
 public:
  TRTBackend(const InferenceOptions &options, const std::shared_ptr<ILogger> &logger);
  virtual ~TRTBackend();

  std::unordered_map<std::string, std::vector<uint8_t>> doInference(
      const std::unordered_map<std::string, std::pair<uint8_t *, size_t>> &inputs) override;

 private:
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  const std::shared_ptr<ILogger> logger_;

  std::vector<void *> buffers_;
  std::vector<size_t> buf_bytes_;
  const std::vector<int> output_indices;
};
}  // namespace ssdf::serve::trt
