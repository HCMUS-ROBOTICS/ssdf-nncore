#include "serve/inference.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "serve/logger.hpp"

namespace ssdf::serve {
Factory<IBackend, std::string,
        std::function<std::unique_ptr<IBackend>(const InferenceOptions &,
                                                const std::shared_ptr<ILogger> &)>>
    backend_factory;

Session::Session(const InferenceOptions &options, const std::shared_ptr<ILogger> &logger)
    : backend_(backend_factory.createObject(options.model_path.extension(), options, logger)) {}

std::unordered_map<std::string, std::vector<uint8_t>> Session::doInference(
    const std::unordered_map<std::string, std::pair<uint8_t *, size_t>> &inputs) {
  return backend_->doInference(inputs);
}
}  // namespace ssdf::serve
