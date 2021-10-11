#include "trt/calibrator.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <random>

#include "serve/device.hpp"

namespace ssdf::serve::trt {
RndInt8Calibrator::RndInt8Calibrator(int batches, const std::filesystem::path& cache_file,
                                     const ILogger& logger,
                                     const nvinfer1::INetworkDefinition& network,
                                     std::vector<int64_t>* elem_count)
    : batches_(batches), current_batch_(0), cache_file_(cache_file), logger_(logger) {
  std::ifstream tryCache(cache_file, std::ios::binary);
  if (tryCache.good()) {
    return;
  }

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  auto gen = [&generator, &distribution]() { return distribution(generator); };

  for (int i = 0; i < network.getNbInputs(); i++) {
    auto* input = network.getInput(i);
    std::vector<float> rnd_data(elem_count->at(i));
    std::generate_n(rnd_data.begin(), elem_count->at(i), gen);

    void* data;
    cudaCheck(cudaMalloc(&data, elem_count->at(i) * sizeof(float)), logger_);
    cudaCheck(cudaMemcpy(data, rnd_data.data(), elem_count->at(i) * sizeof(float),
                         cudaMemcpyHostToDevice),
              logger_);

    device_input_buffers_.insert(std::make_pair(input->getName(), data));
  }
}

RndInt8Calibrator::~RndInt8Calibrator() {
  for (auto& elem : device_input_buffers_) {
    cudaCheck(cudaFree(elem.second), logger_);
  }
}

bool RndInt8Calibrator::getBatch(void* bindings[], const char* names[], int nb_bingdinds) noexcept {
  if (current_batch_ >= batches_) {
    return false;
  }

  for (int i = 0; i < nb_bingdinds; ++i) {
    bindings[i] = device_input_buffers_[names[i]];
  }

  ++current_batch_;

  return true;
}

int RndInt8Calibrator::getBatchSize() const noexcept { return 1; }

const void* RndInt8Calibrator::readCalibrationCache(size_t& length) noexcept {
  calibration_cache_.clear();
  std::ifstream input(cache_file_, std::ios::binary);
  input >> std::noskipws;
  if (input.good()) {
    std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
              std::back_inserter(calibration_cache_));
  }

  length = calibration_cache_.size();
  return !calibration_cache_.empty() ? calibration_cache_.data() : nullptr;
}
}  // namespace ssdf::serve::trt
