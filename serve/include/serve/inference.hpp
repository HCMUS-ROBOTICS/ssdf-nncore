#pragma once
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "serve/factory.hpp"
#include "serve/logger.hpp"

namespace ssdf::serve {
struct InferenceOptions {
  std::filesystem::path model_path;
};

class IBackend {
 public:
  explicit IBackend(const InferenceOptions &options) {}

  virtual std::unordered_map<std::string, std::vector<uint8_t>> doInference(
      const std::unordered_map<std::string, std::pair<uint8_t *, size_t>> &inputs) = 0;
};

class Session {
 public:
  explicit Session(const InferenceOptions &options, const std::shared_ptr<ILogger> &logger);

  std::unordered_map<std::string, std::vector<uint8_t>> doInference(
      const std::unordered_map<std::string, std::pair<uint8_t *, size_t>> &inputs);

 private:
  std::unique_ptr<IBackend> backend_;
};

extern Factory<IBackend, std::string,
               std::function<std::unique_ptr<IBackend>(const InferenceOptions &,
                                                       const std::shared_ptr<ILogger> &)>>
    backend_factory;
}  // namespace ssdf::serve
