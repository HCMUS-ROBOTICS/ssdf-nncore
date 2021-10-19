#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include "serve/device.hpp"
#include "serve/logger.hpp"
#include "trt/option.hpp"
#include "trt/utils.hpp"

namespace ssdf::serve::trt {
/**
 * @brief Class to handle TensorRT engine construction
 *
 */
class Generator {
 public:
  Generator(const BuildOptions& build, const SystemOptions& system,
            const std::shared_ptr<ILogger>& logger)
      : build_(build), system_(system), logger_(logger) {}

  /**
   * @brief Get the Serialized Engine object
   *
   * @tparam network_t
   * @param network_def Model definition, could be file path or function which uses TensorRT API to
   * populate INetworkDefinition
   * @return nvinfer1::IHostMemory*
   */
  template <typename network_t>
  nvinfer1::IHostMemory* getSerializedEngine(const network_t& network_def) {
    static_assert(std::is_convertible_v<network_t, std::filesystem::path>);
    std::vector<std::vector<char>> sparse_weights;  // TODO(Ky) move to inside function?
    auto&& [builder, network, config] = this->setupBuildEnvironment(&sparse_weights);

    std::unique_ptr<nvonnxparser::IParser> parser{nullptr};
    if constexpr (std::is_convertible_v<network_t, std::filesystem::path>) {
      const std::filesystem::path& onnx_path{network_def};
      parser.reset(nvonnxparser::createParser(*network, logger_->getTRTLogger()));
      if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(logger_->getCurrentLevel()))) {
        logger_->fatal("Failed to parse onnx file");
        return nullptr;
      }
    } else {
      throw std::invalid_argument("Engine from network definition is not implemented");
      // TODO(Ky) Network definition
    }

    std::unique_ptr<nvinfer1::ITimingCache> timing_cache{nullptr};
    // Try to load cache from file. Create a fresh cache if the file doesn't exist
    if (build_.timing_cache_mode == TimingCacheMode::kGLOBAL) {
      auto loaded_cache = loadTimingCacheFile(build_.timing_cache_file, *logger_);
      timing_cache.reset(
          config->createTimingCache(static_cast<void*>(loaded_cache.data()), loaded_cache.size()));
      if (!timing_cache) {
        logger_->fatal("TimingCache creation failed");
        return nullptr;
      }
      config->setTimingCache(*timing_cache, false);
    }

    // CUDA stream used for profiling by the builder.
    auto profile_stream = makeCudaStream();
    if (!profile_stream) {
      logger_->fatal("Cuda stream creation failed");
      return nullptr;
    }
    config->setProfileStream(*profile_stream);

    return builder->buildSerializedNetwork(*network, *config);
  }

 private:
  using BuildEnvironment =
      std::tuple<std::unique_ptr<nvinfer1::IBuilder>, std::unique_ptr<nvinfer1::INetworkDefinition>,
                 std::unique_ptr<nvinfer1::IBuilderConfig>>;

  /**
   * @brief Base on TensorRT's networkToSerialized()
   *
   * @param sparse_weights
   * @return BuildEnvironment
   */
  BuildEnvironment setupBuildEnvironment(std::vector<std::vector<char>>* sparse_weights);

  BuildOptions build_;
  SystemOptions system_;
  const std::shared_ptr<ILogger> logger_;
};
}  // namespace ssdf::serve::trt
