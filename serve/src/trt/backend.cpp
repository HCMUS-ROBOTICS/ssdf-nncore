#include "trt/backend.hpp"

#include <NvInfer.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "serve/device.hpp"
#include "serve/inference.hpp"
#include "serve/logger.hpp"
#include "trt/utils.hpp"

namespace ssdf::serve::trt {
namespace {
const bool registered = backend_factory.doRegister(
    {".engine", ".trt"},
    [](const InferenceOptions &options, const std::shared_ptr<ILogger> &logger) {
      return std::make_unique<TRTBackend>(options, logger);
    });

nvinfer1::ICudaEngine *loadEngine(const std::filesystem::path &model_path,
                                  const std::shared_ptr<ILogger> &logger) {
  std::ifstream engine_file(model_path, std::ios::binary);
  if (!engine_file) {
    // TODO(all) remove c_str() when std::format is supported
    logger->fatal("Error opening engine file: {}", model_path.c_str());
    return nullptr;
  }
  uint64_t file_size = std::filesystem::file_size(model_path);
  std::vector<char> engine_data(file_size);
  engine_file.read(engine_data.data(), file_size);
  if (!engine_file) {
    logger->fatal("Error loading engine file: {}", model_path.c_str());
    return nullptr;
  }
  std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger->getTRTLogger())};
  // TODO(Ky) runtime->setErrorRecorder(&gRecorder);
  return runtime->deserializeCudaEngine(engine_data.data(), file_size);
}

std::vector<int> getOutputIndices(const nvinfer1::ICudaEngine &engine) {
  std::vector<int> indices;
  for (int i = 0; i < engine.getNbBindings(); ++i) {
    if (!engine.bindingIsInput(i)) {
      indices.emplace_back(i);
    }
  }
  return indices;
}
}  // namespace
TRTBackend::TRTBackend(const InferenceOptions &options, const std::shared_ptr<ILogger> &logger)
    : IBackend(options),
      engine_(loadEngine(options.model_path, logger)),
      context_(engine_->createExecutionContext()),
      logger_(logger),
      output_indices(getOutputIndices(*engine_)) {
  for (int i = 0; i < engine_->getNbBindings(); ++i) {
    // Calculate layer's size
    nvinfer1::Dims dims = context_->getBindingDimensions(i);
    nvinfer1::DataType data_type = engine_->getBindingDataType(i);
    size_t count = 1;
    int vec_dim = engine_->getBindingVectorizedDim(i);
    if (vec_dim != -1) {  // i.e., 0 != lgScalarsPerVector
      int scalarsPerVec = engine_->getBindingComponentsPerElement(i);
      dims.d[vec_dim] = divUp(dims.d[vec_dim], scalarsPerVec);
      count *= scalarsPerVec;
    }
    count *= volume(dims);

    // Allocate device memory
    void *d_ptr;
    size_t layer_bytes = count * getElementSize(data_type);
    cudaCheck(cudaMalloc(&d_ptr, layer_bytes), *logger_);
    buffers_.emplace_back(d_ptr);
    buf_bytes_.emplace_back(layer_bytes);
  }
}

TRTBackend::~TRTBackend() {
  for (auto d_ptr : buffers_) {
    cudaCheck(cudaFree(d_ptr), *logger_);
  }
}

std::unordered_map<std::string, std::vector<uint8_t>> TRTBackend::doInference(
    const std::unordered_map<std::string, std::pair<uint8_t *, size_t>> &inputs) {
  // Copy inputs to device
  for (const auto &[name, input] : inputs) {
    int index = engine_->getBindingIndex(name.c_str());
    if (index == -1) {
      throw std::invalid_argument("Binding '" + name + "' is not found");
    }
    if (buf_bytes_[index] != input.second) {
      throw std::invalid_argument("Invalid input size");
    }
    cudaCheck(cudaMemcpy(buffers_[index], input.first, buf_bytes_[index], cudaMemcpyHostToDevice),
              *logger_);
  }

  // Forward
  context_->executeV2(buffers_.data());

  // Copy outputs to host
  std::unordered_map<std::string, std::vector<uint8_t>> outputs;
  for (int index : output_indices) {
    std::vector<uint8_t> output(buf_bytes_[index]);
    // Not throw because outputs is constructed from engine
    cudaCheck(cudaMemcpy(output.data(), buffers_[index], buf_bytes_[index], cudaMemcpyDeviceToHost),
              *logger_);
    outputs.emplace(engine_->getBindingName(index), std::move(output));
  }
  return outputs;
}
}  // namespace ssdf::serve::trt
