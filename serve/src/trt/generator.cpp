#include "trt/generator.hpp"

#include <fmt/ranges.h>

#include <fstream>
#include <unordered_map>
#include <utility>

#include "half.hpp"
#include "trt/calibrator.hpp"
#include "trt/option.hpp"
#include "trt/utils.hpp"

namespace ssdf::serve::trt {
namespace {
std::unordered_map<std::string, float> readScalesFromCalibrationCache(
    const std::filesystem::path& calibration_file, const ILogger& logger) {
  std::unordered_map<std::string, float> tensor_scales;
  std::ifstream cache{calibration_file};
  if (!cache) {
    logger.error("[TRT] Can not open provided calibration cache file");
    return tensor_scales;
  }
  std::string line;
  while (std::getline(cache, line)) {
    auto colon_pos = line.find_last_of(':');
    if (colon_pos != std::string::npos) {
      // Scales should be stored in calibration cache as 32-bit floating numbers
      // encoded as 32-bit integers
      int32_t scales_as_int = std::stoi(line.substr(colon_pos + 2, 8), nullptr, 16);
      const auto tensor_name = line.substr(0, colon_pos);
      tensor_scales.emplace(tensor_name, *reinterpret_cast<float*>(&scales_as_int));
    }
  }
  return tensor_scales;
}

bool setTensorDynamicRange(const nvinfer1::INetworkDefinition& network, float in_range = 2.0f,
                           float out_range = 4.0f) {
  // Ensure that all layer inputs have a dynamic range.
  for (int l = 0; l < network.getNbLayers(); ++l) {
    nvinfer1::ILayer* layer{network.getLayer(l)};
    for (int i = 0; i < layer->getNbInputs(); ++i) {
      nvinfer1::ITensor* input{layer->getInput(i)};
      // Optional inputs are nullptr here and are from RNN layers.
      if (input && !input->dynamicRangeIsSet()) {
        if (!input->setDynamicRange(-in_range, in_range)) {
          return false;
        }
      }
    }
    for (int o = 0; o < layer->getNbOutputs(); ++o) {
      nvinfer1::ITensor* output{layer->getOutput(o)};
      // Optional outputs are nullptr here and are from RNN layers.
      if (output && !output->dynamicRangeIsSet()) {
        // Pooling must have the same input and output dynamic range.
        if (layer->getType() == nvinfer1::LayerType::kPOOLING) {
          if (!output->setDynamicRange(-in_range, in_range)) {
            return false;
          }
        } else {
          if (!output->setDynamicRange(-out_range, out_range)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

void setTensorScalesFromCalibration(const std::vector<IOFormat>& input_formats,
                                    const std::vector<IOFormat>& output_formats,
                                    const std::filesystem::path& calibration_file,
                                    const nvinfer1::INetworkDefinition& network,
                                    const ILogger& logger) {
  const auto tensor_scales = readScalesFromCalibrationCache(calibration_file, logger);
  const bool broadcast_input_formats = broadcastIOFormats(input_formats, network.getNbInputs());
  for (int32_t i = 0, n = network.getNbInputs(); i < n; ++i) {
    int32_t formatIdx = broadcast_input_formats ? 0 : i;
    if (!input_formats.empty() && input_formats[formatIdx].first == nvinfer1::DataType::kINT8) {
      auto* input = network.getInput(i);
      const auto calib_scale = tensor_scales.at(input->getName());
      input->setDynamicRange(-127 * calib_scale, 127 * calib_scale);
    }
  }
  const bool broadcast_output_formats = broadcastIOFormats(output_formats, network.getNbInputs());
  for (int32_t i = 0, n = network.getNbOutputs(); i < n; ++i) {
    int32_t format_idx = broadcast_output_formats ? 0 : i;
    if (!output_formats.empty() && output_formats[format_idx].first == nvinfer1::DataType::kINT8) {
      auto* output = network.getOutput(i);
      const auto calib_scale = tensor_scales.at(output->getName());
      output->setDynamicRange(-127 * calib_scale, 127 * calib_scale);
    }
  }
}

template <typename T>
void sparsify(const T* values, int64_t count, int32_t k, int32_t rs,
              std::vector<char>* sparse_weights) {
  const auto c = count / (k * rs);
  sparse_weights->resize(count * sizeof(T));
  auto* sparse_values = reinterpret_cast<T*>(sparse_weights->data());

  constexpr int32_t window = 4;
  constexpr int32_t nonzeros = 2;

  const int32_t crs = c * rs;
  const auto getIndex = [=](int32_t ki, int32_t ci, int32_t rsi) {
    return ki * crs + ci * rs + rsi;
  };

  for (int64_t ki = 0; ki < k; ++ki) {
    for (int64_t rsi = 0; rsi < rs; ++rsi) {
      int32_t w = 0;
      int32_t nz = 0;
      for (int64_t ci = 0; ci < c; ++ci) {
        const auto index = getIndex(ki, ci, rsi);
        if (nz < nonzeros) {
          sparse_values[index] = values[index];
          ++nz;
        } else {
          sparse_values[index] = 0;
        }
        if (++w == window) {
          w = 0;
          nz = 0;
        }
      }
    }
  }
}

void sparsify(const nvinfer1::Weights& weights, int32_t k, int32_t rs,
              std::vector<char>* sparse_weights) {
  using nvinfer1::DataType;
  switch (weights.type) {
    case DataType::kFLOAT:
      sparsify(static_cast<const float*>(weights.values), weights.count, k, rs, sparse_weights);
      break;
    case DataType::kHALF:
      sparsify(static_cast<const half_float::half*>(weights.values), weights.count, k, rs,
               sparse_weights);
      break;
    case DataType::kINT8:
    case DataType::kINT32:
    case DataType::kBOOL:
      break;
  }
}

template <typename Layer>
void setSparseWeights(int32_t k, int32_t rs, Layer* layer, std::vector<char>* sparse_weights) {
  auto weights = layer->getKernelWeights();
  sparsify(weights, k, rs, sparse_weights);
  weights.values = sparse_weights->data();
  layer->setKernelWeights(weights);
}

void sparsify(const nvinfer1::INetworkDefinition& network,
              std::vector<std::vector<char>>* sparse_weights) {
  for (int32_t l = 0; l < network.getNbLayers(); ++l) {
    nvinfer1::ILayer* layer = network.getLayer(l);
    const auto layer_type = layer->getType();
    if (layer_type == nvinfer1::LayerType::kCONVOLUTION) {
      auto& conv = *static_cast<nvinfer1::IConvolutionLayer*>(layer);
      const auto& dims = conv.getKernelSizeNd();
      if (dims.nbDims > 2) {
        continue;
      }
      const auto k = conv.getNbOutputMaps();
      const auto rs = dims.d[0] * dims.d[1];
      sparse_weights->emplace_back();
      setSparseWeights(k, rs, &conv, &sparse_weights->back());
    } else if (layer_type == nvinfer1::LayerType::kFULLY_CONNECTED) {
      auto& fc = *static_cast<nvinfer1::IFullyConnectedLayer*>(layer);
      const auto k = fc.getNbOutputChannels();
      sparse_weights->emplace_back();
      setSparseWeights(k, 1, &fc, &sparse_weights->back());
    }
  }
}

void setConfigFlags(const BuildOptions& build,
                    const std::unique_ptr<nvinfer1::IBuilderConfig>& config) {
  using nvinfer1::BuilderFlag;
  config->setMaxWorkspaceSize(static_cast<size_t>(build.workspace_mb) << 20);
  if (build.timing_cache_mode == TimingCacheMode::kDISABLE) {
    config->setFlag(BuilderFlag::kDISABLE_TIMING_CACHE);
  }
  if (!build.tf32) {
    config->clearFlag(BuilderFlag::kTF32);
  }
  if (build.refittable) {
    config->setFlag(BuilderFlag::kREFIT);
  }
  config->setProfilingVerbosity(build.profiling_verbosity);
  if (build.min_timing > 0) {
    config->setMinTimingIterations(build.min_timing);
  }
  if (build.avg_timing > 0) {
    config->setAvgTimingIterations(build.avg_timing);
  }
  if (build.fp16) {
    config->setFlag(BuilderFlag::kFP16);
  }
  if (build.int8) {
    config->setFlag(BuilderFlag::kINT8);
  }
  if (build.strict_types) {
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
  }
  if (build.restricted) {
    config->setFlag(BuilderFlag::kSAFETY_SCOPE);
  }
  if (build.enabled_tactics || build.disabled_tactics) {
    nvinfer1::TacticSources tacticSources = config->getTacticSources();
    tacticSources |= build.enabled_tactics;
    tacticSources &= ~build.disabled_tactics;
    config->setTacticSources(tacticSources);
  }
}
}  // namespace

Generator::BuildEnvironment Generator::setupBuildEnvironment(
    std::vector<std::vector<char>>* sparse_weights) {
  std::unique_ptr<nvinfer1::IBuilder> builder{
      nvinfer1::createInferBuilder(logger_->getTRTLogger())};
  if (!builder) {
    throw std::runtime_error("Builder creation failed");
  }

  std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(
      build_.max_batch
          ? 0U
          : 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH))};
  if (!network) {
    throw std::runtime_error("Network creation failed");
  }

  nvinfer1::IOptimizationProfile* profile = nullptr;
  if (build_.max_batch) {
    builder->setMaxBatchSize(build_.max_batch);
  } else {
    profile = builder->createOptimizationProfile();
  }

  bool has_dynamic_shapes = false;
  bool broadcast_input_formats = broadcastIOFormats(build_.input_formats, network->getNbInputs());
  // Check if the provided input tensor names match the input tensors of the
  // engine. Throw an error if the provided input tensor names cannot be found
  // because it implies a potential typo.
  if (profile) {
    for (const auto& shape : build_.shapes) {
      bool tensor_name_found = false;
      for (int32_t i = 0; i < network->getNbInputs(); ++i) {
        if (network->getInput(i)->getName() == shape.first) {
          tensor_name_found = true;
          break;
        }
      }
      if (!tensor_name_found) {
        throw std::invalid_argument(
            "Cannot find input tensor with name \"" + shape.first +
            "\" in the network inputs! Please make sure the input tensor names are correct.");
      }
    }
  }

  using nvinfer1::DataType, nvinfer1::OptProfileSelector;
  // Set formats and data types of inputs
  for (uint32_t i = 0, n = network->getNbInputs(); i < n; ++i) {
    nvinfer1::ITensor* input = network->getInput(i);
    if (!build_.input_formats.empty()) {
      int input_format_index = broadcast_input_formats ? 0 : i;
      input->setType(build_.input_formats[input_format_index].first);
      input->setAllowedFormats(build_.input_formats[input_format_index].second);
    } else {
      switch (input->getType()) {
        case DataType::kINT32:
        case DataType::kBOOL:
        case DataType::kHALF:
          // Leave these as is.
          break;
        case DataType::kFLOAT:
        case DataType::kINT8:
          // User did not specify a floating-point format. Default to kFLOAT.
          input->setType(DataType::kFLOAT);
          break;
      }
      input->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
    }

    if (profile) {
      nvinfer1::Dims dims = input->getDimensions();
      const bool isScalar = dims.nbDims == 0;
      const bool is_dynamic_input =
          std::any_of(dims.d, dims.d + dims.nbDims, [](int dim) { return dim == -1; }) ||
          input->isShapeTensor();
      if (is_dynamic_input) {
        has_dynamic_shapes = true;
        auto shape = build_.shapes.find(input->getName());
        ShapeRange shapes{};

        // If no shape is provided, set dynamic dimensions to 1.
        if (shape == build_.shapes.end()) {
          constexpr int DEFAULT_DIMENSION = 1;
          std::vector<int> static_dims;
          if (input->isShapeTensor()) {
            if (isScalar) {
              static_dims.push_back(1);
            } else {
              static_dims.resize(dims.d[0]);
              std::fill(static_dims.begin(), static_dims.end(), DEFAULT_DIMENSION);
            }
          } else {
            static_dims.resize(dims.nbDims);
            std::transform(dims.d, dims.d + dims.nbDims, static_dims.begin(), [&](int dimension) {
              return dimension > 0 ? dimension : DEFAULT_DIMENSION;
            });
          }
          logger_->warn(
              "Dynamic dimensions required for input: {}, but no shapes were provided. "
              "Automatically overriding shape to: {}",
              input->getName(), fmt::join(static_dims, "x"));
          std::fill(shapes.begin(), shapes.end(), static_dims);
        } else {
          shapes = shape->second;
        }

        std::vector<int> profile_dims;
        if (input->isShapeTensor()) {
          const auto setShapeValues = [&](OptProfileSelector selector, const std::string& name) {
            profile_dims = shapes[static_cast<size_t>(selector)];
            if (!profile->setShapeValues(input->getName(), selector, profile_dims.data(),
                                         static_cast<int>(profile_dims.size()))) {
              throw std::runtime_error("Error in set shape values " + name);
            }
            return true;
          };
          setShapeValues(OptProfileSelector::kMIN, "MIN");
          setShapeValues(OptProfileSelector::kOPT, "OPT");
          setShapeValues(OptProfileSelector::kMAX, "MAX");
        } else {
          const auto setDimensions = [&](OptProfileSelector selector, const std::string& name) {
            profile_dims = shapes[static_cast<size_t>(selector)];
            if (!profile->setDimensions(input->getName(), selector,
                                        toDims(profile_dims, *logger_))) {
              throw std::runtime_error("Error in set dimensions to profile " + name);
            }
            return true;
          };
          setDimensions(OptProfileSelector::kMIN, "MIN");
          setDimensions(OptProfileSelector::kOPT, "OPT");
          setDimensions(OptProfileSelector::kMAX, "MAX");
        }
      }
    }
  }

  if (!has_dynamic_shapes && !build_.shapes.empty()) {
    throw std::invalid_argument(
        "Static model does not take explicit shapes since the shape of inference tensors will be "
        "determined by the model itself");
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
  if (profile && has_dynamic_shapes) {
    if (!profile->isValid()) {
      throw std::runtime_error("Required optimization profile is invalid");
    }
    if (config->addOptimizationProfile(profile) == -1) {
      throw std::runtime_error("Error in add optimization profile");
    }
  }

  bool broadcast_output_formats =
      broadcastIOFormats(build_.output_formats, network->getNbOutputs(), false);
  for (uint32_t i = 0, n = network->getNbOutputs(); i < n; ++i) {
    // Set formats and data types of outputs
    nvinfer1::ITensor* output = network->getOutput(i);
    if (!build_.output_formats.empty()) {
      int outputFormatIndex = broadcast_output_formats ? 0 : i;
      output->setType(build_.output_formats[outputFormatIndex].first);
      output->setAllowedFormats(build_.output_formats[outputFormatIndex].second);
    } else {
      output->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
    }
  }

  setConfigFlags(build_, config);
  if (build_.sparsity != SparsityFlag::kDISABLE) {
    config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
    if (build_.sparsity == SparsityFlag::kFORCE) {
      sparsify(*network, sparse_weights);
    }
  }
  if (build_.int8 && !build_.fp16) {
    logger_->info(
        "FP32 and INT8 precisions have been specified - consider enable FP16 for more "
        "performance");
  }

  auto isInt8 = [](const IOFormat& format) { return format.first == DataType::kINT8; };
  auto int8IO = std::count_if(build_.input_formats.begin(), build_.input_formats.end(), isInt8) +
                std::count_if(build_.output_formats.begin(), build_.output_formats.end(), isInt8);
  auto hasQDQLayers = [](const nvinfer1::INetworkDefinition& network) {
    // Determine if our network has QDQ layers.
    const auto nb_layers = network.getNbLayers();
    for (int32_t i = 0; i < nb_layers; ++i) {
      const auto& layer = network.getLayer(i);
      if (layer->getType() == nvinfer1::LayerType::kQUANTIZE ||
          layer->getType() == nvinfer1::LayerType::kDEQUANTIZE) {
        return true;
      }
    }
    return false;
  };

  if (!hasQDQLayers(*network) && (build_.int8 || int8IO) && build_.calibration.empty()) {
    // Explicitly set int8 scales if no calibrator is provided and if I/O tensors use int8,
    // because auto calibration does not support this case.
    if (!setTensorDynamicRange(*network)) {
      throw std::runtime_error("Error in set tensor dynamic range.");
    }
  } else if (build_.int8) {
    if (!hasQDQLayers(*network) && int8IO) {
      try {
        // Set dynamic ranges of int8 inputs / outputs to match scales loaded
        // from calibration cache
        // TODO(nvidia) http://nvbugs/3262234 Change the network validation so that this
        // workaround can be removed
        setTensorScalesFromCalibration(build_.input_formats, build_.output_formats,
                                       build_.calibration, *network, *logger_);
      } catch (const std::exception&) {
        throw std::runtime_error(
            "Int8IO was specified but impossible to read tensor scales from provided calibration "
            "cache file");
      }
    }

    nvinfer1::IOptimizationProfile* profile_calib = nullptr;
    if (!build_.shapes_calib.empty()) {
      profile_calib = builder->createOptimizationProfile();
      for (uint32_t i = 0, n = network->getNbInputs(); i < n; ++i) {
        auto* input = network->getInput(i);
        nvinfer1::Dims profile_dims{};
        auto shape = build_.shapes_calib.find(input->getName());
        ShapeRange shapes_calib{};
        shapes_calib = shape->second;

        profile_dims =
            toDims(shapes_calib[static_cast<size_t>(OptProfileSelector::kOPT)], *logger_);
        // Here we check only kMIN as all profileDims are the same.
        if (!profile_calib->setDimensions(input->getName(), OptProfileSelector::kMIN,
                                          profile_dims)) {
          throw std::runtime_error("Error in set dimensions to calibration profile OPT");
        }
        profile_calib->setDimensions(input->getName(), OptProfileSelector::kOPT, profile_dims);
        profile_calib->setDimensions(input->getName(), OptProfileSelector::kMAX, profile_dims);
      }
      if (!profile_calib->isValid()) {
        throw std::runtime_error("Calibration profile is invalid");
      }
      if (!config->setCalibrationProfile(profile_calib)) {
        throw std::runtime_error("Error in set calibration profile");
      }
    }

    std::vector<int64_t> elem_count{};
    for (int i = 0; i < network->getNbInputs(); ++i) {
      auto* input = network->getInput(i);
      if (profile_calib) {
        elem_count.push_back(
            volume(profile_calib->getDimensions(input->getName(), OptProfileSelector::kOPT)));
      } else if (profile && has_dynamic_shapes) {
        elem_count.push_back(
            volume(profile->getDimensions(input->getName(), OptProfileSelector::kOPT)));
      } else {
        elem_count.push_back(volume(input->getDimensions()));
      }
    }
    config->setInt8Calibrator(
        new RndInt8Calibrator(1, build_.calibration, logger_, *network, &elem_count));
  }

  if (build_.safe) {
    config->setEngineCapability(system_.dla_core != -1 ? nvinfer1::EngineCapability::kDLA_STANDALONE
                                                       : nvinfer1::EngineCapability::kSAFETY);
  }

  if (system_.dla_core != -1) {
    if (system_.dla_core < builder->getNbDLACores()) {
      config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
      config->setDLACore(system_.dla_core);
      config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
      if (system_.fallback) {
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
      }
      if (!build_.int8) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
      }
    } else {
      throw std::invalid_argument("Cannot create DLA engine, " + std::to_string(system_.dla_core) +
                                  " not available");
    }
  }
  return std::make_tuple(std::move(builder), std::move(network), std::move(config));
}
}  // namespace ssdf::serve::trt
