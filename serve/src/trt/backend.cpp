#include "trt/backend.hpp"

#include <cassert>
#include <fstream>

#include "serve/device.hpp"
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
}  // namespace

TRTBackend::TRTBackend(const InferenceOptions &options, const std::shared_ptr<ILogger> &logger)
    : IBackend(options),
      engine_(loadEngine(options.model_path, logger)),
      context_(engine_->createExecutionContext()),
      logger_(logger) {
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
    size_t layer_bytes = count * getElementSize(data_type);

    // Allocate device memory
    void *d_ptr;
    if (cudaMalloc(&d_ptr, layer_bytes) != cudaSuccess) {
      throw std::bad_alloc();
    }
    d_ptrs_.emplace_back(d_ptr);
    d_bytes_.emplace_back(layer_bytes);

    // Allocate outputs map
    if (!engine_->bindingIsInput(i)) {
      d_ptr = new char[d_bytes_[i]];
      if (!d_ptr) {
        throw std::bad_alloc();
      }
      outputs_.emplace(engine_->getBindingName(i), d_ptr);
    }
  }
}

TRTBackend::~TRTBackend() {
  for (void *d_ptr : d_ptrs_) {
    cudaFree(d_ptr);
  }
  for (auto &[name, output] : outputs_) {
    delete[] static_cast<char *>(output);
  }
}

std::unordered_map<std::string, void *> TRTBackend::doInference(
    const std::unordered_map<std::string, void *> &inputs) {
  // Copy inputs to device
  for (auto const &[name, h_ptr] : inputs) {
    int index = engine_->getBindingIndex(name.c_str());
    if (index == -1) {
      throw std::invalid_argument("Binding \"" + name + "\" is not found");
    }
    cudaCheck(cudaMemcpy(d_ptrs_[index], h_ptr, d_bytes_[index], cudaMemcpyHostToDevice), *logger_);
  }

  // Forward
  context_->executeV2(d_ptrs_.data());

  // Copy outputs to host
  for (auto &[name, h_ptr] : outputs_) {
    int index = engine_->getBindingIndex(name.c_str());
    // Not throw because outputs is constructed from engine
    cudaCheck(cudaMemcpy(h_ptr, d_ptrs_[index], d_bytes_[index], cudaMemcpyDeviceToHost), *logger_);
  }
  return outputs_;
}
}  // namespace ssdf::serve::trt
