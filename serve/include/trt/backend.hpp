#pragma once
#include <NvInfer.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "serve/inference.hpp"
#include "serve/logger.hpp"

namespace ssdf::serve::trt {
class TRTBackend : public IBackend {
 public:
  TRTBackend(const InferenceOptions &options, const std::shared_ptr<ILogger> &logger);
  virtual ~TRTBackend();

  std::unordered_map<std::string, void *> doInference(
      const std::unordered_map<std::string, void *> &inputs) override;

 private:
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  const std::shared_ptr<ILogger> logger_;

  std::vector<void *> d_ptrs_;
  std::vector<size_t> d_bytes_;
  std::unordered_map<std::string, void *> outputs_;
};
}  // namespace ssdf::serve::trt
