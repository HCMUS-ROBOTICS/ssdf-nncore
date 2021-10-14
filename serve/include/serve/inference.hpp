#pragma once
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "serve/factory.hpp"
#include "serve/logger.hpp"

namespace ssdf::serve {
struct InferenceOptions {
  std::filesystem::path model_path;
};

class IBackend {
 public:
  explicit IBackend(const InferenceOptions &options) {}

  virtual std::unordered_map<std::string, void *> doInference(
      const std::unordered_map<std::string, void *> &inputs) = 0;
};

class Session {
 public:
  explicit Session(const InferenceOptions &options);

  std::unordered_map<std::string, void *> doInference(
      const std::unordered_map<std::string, void *> &inputs);

 private:
  std::unique_ptr<IBackend> backend_;
};

extern Factory<IBackend, std::string,
               std::function<std::unique_ptr<IBackend>(const InferenceOptions &,
                                                       const std::shared_ptr<ILogger> &)>>
    backend_factory;
}  // namespace ssdf::serve
