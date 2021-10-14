#include "serve/inference.hpp"

namespace ssdf::serve {
Factory<IBackend, std::string,
        std::function<std::unique_ptr<IBackend>(const InferenceOptions &,
                                                const std::shared_ptr<ILogger> &)>>
    backend_factory;

Session::Session(const InferenceOptions &options)
    : backend_(backend_factory.createObject(options.model_path.extension())) {}

std::unordered_map<std::string, void *> Session::doInference(
    const std::unordered_map<std::string, void *> &inputs) {
  return backend_->doInference(inputs);
}
}  // namespace ssdf::serve
